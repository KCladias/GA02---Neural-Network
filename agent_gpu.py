"""
store all the agents here
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json


def huber_loss(y_true, y_pred, delta=1.0):
    """PyTorch implementation for huber loss."""
    error = y_true - y_pred
    condition = torch.abs(error) < delta
    quad_error = 0.5 * torch.square(error)
    lin_error = delta * (torch.abs(error) - 0.5 * delta)
    loss = torch.where(condition, quad_error, lin_error)
    return loss

def mean_huber_loss(y_true, y_pred, delta=1.0):
    """Calculates the mean value of huber loss in PyTorch."""
    return torch.mean(huber_loss(y_true, y_pred, delta))



class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent
    HamiltonianCycleAgent
    BreadthFirstSearchAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _n_frames : int
        Total frames to keep in history when making prediction
        should be the same as env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _board_grid : Numpy array
        A square filled with values from 0 to board size **2,
        Useful when converting between row, col and int representation
    _version : str
        model version string
    """
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """ initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        frames : int, optional
            The env frame count to keep old frames in state
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2)\
                             .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                    self._n_frames, self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point//self._board_size, point%self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row*self._board_size + col
    
class DeepQLearningAgent(Agent):
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    """
    def __init__(self, board_size=10, frames=4, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        """Initializer for DQN agent, arguments are same as Agent class
        except use_target_net is by default True and we call and additional
        reset models method to initialize the DQN networks
        """
        Agent.__init__(self, board_size=board_size, frames=frames, buffer_size=buffer_size,
                 gamma=gamma, n_actions=n_actions, use_target_net=use_target_net,
                 version=version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset_models()
        self.optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model().to(self.device)
        if(self._use_target_net):
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the PyTorch DQN model.

        Parameters
        ----------
        board : Numpy array or Tensor
            The board state for which to predict action values.
        model : PyTorch Model, optional
            The model to use for prediction, default to self._model if not specified.

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions.
        """
        # Check if board is a Tensor, if not, convert it
        if not isinstance(board, torch.Tensor):
            board = torch.from_numpy(board).float()
            board = board.to(self.device)

        # Use the specified model or default to self._model
        if model is None:
            model = self._model.to(self.device)

        # Ensure the model is in evaluation mode
        model.eval()

        # Forward pass to get outputs
        with torch.no_grad():
            model_outputs = model(board)

        # Convert outputs back to numpy array if necessary
        if isinstance(model_outputs, torch.Tensor):
            return model_outputs.cpu().numpy().to(self.device)
        else:
            return model_outputs.to(self.device)

    def _prepare_input(self, board):
        """Prepare the input by reshaping and normalizing.

        Parameters
        ----------
        board : Numpy array
            The board state to process.

        Returns
        -------
        board : Numpy array
            Processed and normalized board.
        """
        board = board.to(self.device)
        if board.ndim == 3:
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board).to(self.device)
        # Transpose the board to match PyTorch's input shape
        return board.to(self.device)

    def _normalize_board(self, board):
        # Check if the board is a NumPy array and convert it to a PyTorch tensor if necessary
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board).float().to(self.device)
        
        # Normalize the board
        return board / 4.0


    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves==1, model_outputs, -np.inf), axis=1)

    def _agent_model(self):
        # Load model configuration
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            config = json.loads(f.read())
        
        # PyTorch CNN model
        class CNNModel(nn.Module):
            def __init__(self, config):
                super(CNNModel, self).__init__()
                # Extract configuration details
                conv1_config = config['model']['Conv2D']
                conv2_config = config['model']['Conv2D_1']
                conv3_config = config['model']['Conv2D_2']
                dense_config = config['model']['Dense_1']
                

                # Layers based on configuration
                self.conv1 = nn.Conv2d(in_channels=config['board_size'], out_channels=conv1_config['filters'], kernel_size=tuple(conv1_config['kernel_size']), padding=1)
                self.conv2 = nn.Conv2d(in_channels=conv1_config['filters'], out_channels=conv2_config['filters'], kernel_size=tuple(conv2_config['kernel_size']), padding=1)
                self.conv3 = nn.Conv2d(in_channels=conv2_config['filters'], out_channels=conv3_config['filters'], kernel_size=tuple(conv3_config['kernel_size']), padding=2)


                # Dense layer
                self.fc1 = nn.Linear(1280, dense_config['units'])

                # Output layer
                self.fc2 = nn.Linear(dense_config['units'], config['n_actions'])

            def forward(self, x):
            
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = torch.flatten(x, start_dim=1)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Return an instance of the model
        model = CNNModel(config).to(self.device)
        return model


    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer = self._model.optimizer, 
                            loss = self._model.loss)


    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0

    
        model = self._agent_model().to(self.device)  # Get the model instance
        model_file = f"{file_path}/model_{iteration:04d}.pt"
        model.load_state_dict(torch.load(model_file))
        model = model.to(self.device)
        model.eval()
        return model

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        
        model = self._agent_model().to(self.device)  # Get the model instance
        # Save the model's state dictionary
        torch.save(model.state_dict(), f"{file_path}/model_{iteration:04d}.pt")

        # If using a target network, save its state dict as well
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pt")
    
    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the PyTorch model by sampling from the buffer and return the error.

        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer.
        num_games : int, optional
            Not used here, kept for consistency with other agents.
        reward_clip : bool, optional
            Whether to clip the rewards.

        Returns
        -------
        loss_value : float
            The current loss value.
        """
        # Sample data from the buffer
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        print("1")

        # PyTorch tensors
        s = torch.from_numpy(s).float().to(self.device)
        print("2")
        next_s = torch.from_numpy(next_s).float().to(self.device)
        print("3")
        a = torch.from_numpy(a).long().to(self.device)
        print("4")
        r = torch.from_numpy(r).float().to(self.device)
        print("5")
        done = torch.from_numpy(done).float().to(self.device)
        print("6")


        # Normalize the board states
        s = self._normalize_board(s)
        print("7")
        next_s = self._normalize_board(next_s)
        print("8")

        if reward_clip:
            r = torch.sign(r).to(self.device)
        print("9")


        # Q values for current states
        q_values = self._model(s)
        print("10")

        # Q values for next states using target network
        next_q_values = self._target_net(next_s) if self._use_target_net else self._model(next_s)
        print("11")
        # Maximum Q values for next states
        max_next_q_values = next_q_values.max(1)[0]
        print("12")

        target_q_values = target_q_values.to(self.device)
        print("13")
        max_next_q_values = max_next_q_values.to(self.device)
        print("14")

        # Target Q values for the actions taken
        target_q_values = q_values.clone().to(self.device)  # Clone the current q_values
        print("15")
        for i in range(batch_size):
            target_q_values[i, a[i]] = r[i] + self._gamma * max_next_q_values[i] * (1 - done[i])
            print("16")
        


        # loss
        loss = F.smooth_l1_loss(q_values, target_q_values)
        print("17")
        # Zero gradients, backpropagate, and update the weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def update_target_net(self):
        """Update the weights of the target network in PyTorch."""
        if self._use_target_net:
            self._target_net.load_state_dict(self._model.state_dict())