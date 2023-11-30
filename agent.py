from replay_buffer import ReplayBufferNumpy
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import time
import pickle
from collections import deque
import json



class Agent():
    """Base class for all agents
    This class extends to the following classes
    DeepQLearningAgent

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
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._n_frames, self._board_size, self._board_size)
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2).reshape(self._board_size, -1)
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

    def load_buffer(self, file_path='', iteration=None, return_buffer=False):
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
            print(f'Buffer {file_path}/buffer_{iteration} loaded')
        if return_buffer:
            return self._buffer

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
    """
    This agent class implements the Q-learning algorithm using Deep Learning with PyTorch framework.
    """

    def __init__(self, board_size=10, frames=4, buffer_size=10000,
             gamma=0.99, n_actions=3, use_target_net=True,
             version=''):
        super().__init__(board_size, frames, buffer_size,
                        gamma, n_actions, use_target_net, version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        # Initialize the model
        self._model = self._agent_model().to(self.device)
        if self._use_target_net:
            self._target_net = self._agent_model().to(self.device)
            self.update_target_net()

        # Set up the optimizer and the loss function for training. 
        self._optimizer = optim.RMSprop(self._model.parameters(), lr=0.0005)
        

    def reset_models(self):
        """ Reset all the models """
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()


    def _prepare_input(self, board):
        """
        Prepares the input data for the network. This includes converting the data to a tensor, 
        ensuring it's on the correct device, adjusting its dimensions and normalizing.
        """
        # Convert to a PyTorch tensor
        if not isinstance(board, torch.Tensor):
            board = torch.tensor(board, dtype=torch.float32)
        
        board = board.to(self.device) # Move to device

        # Check if board is a single sample or a batch
        if board.ndim == 3:
            board = board.permute(2, 0, 1).unsqueeze(0)
        elif board.ndim == 4:  
            board = board.permute(0, 3, 1, 2)  # channel-first 

        board = self._normalize_board(board)
        return board

    def _normalize_board(self, board):
        # Normalize the board 
        return board / 4.0


    def _get_model_outputs(self, board, model=None):
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model_outputs = model(board)
        return model_outputs

    
    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value"""
        model_outputs = self._get_model_outputs(board, self._model).to(self.device)
        legal_moves_tensor = torch.tensor(legal_moves).to(self.device)
        # tensor with -np.inf
        infinite_tensor = torch.full_like(model_outputs, -np.inf)

        # Make sure output is on CPU since replay buffer uses Numpy
        return torch.argmax(torch.where(legal_moves_tensor == 1, model_outputs, infinite_tensor), axis=1).cpu().numpy()


    def _agent_model(self):
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            config = json.loads(f.read())
        
        conv1_config = config['model']['Conv2D']
        conv2_config = config['model']['Conv2D_1']
        conv3_config = config['model']['Conv2D_2']
        dense_config = config['model']['Dense_1']
                
        model = nn.Sequential(                                            
            nn.Conv2d(in_channels=config['frames'], out_channels=conv1_config['filters'], kernel_size=tuple(conv1_config['kernel_size']), padding=1),      
            nn.ReLU(),

            nn.Conv2d(in_channels=conv1_config['filters'], out_channels=conv2_config['filters'], kernel_size=tuple(conv2_config['kernel_size'])), 
            nn.ReLU(),                          

            nn.Conv2d(in_channels=conv2_config['filters'], out_channels=conv3_config['filters'], kernel_size=tuple(conv3_config['kernel_size'])), 
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(dense_config['units'], config['n_actions'])
            )
        return model


    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        
        
        # Save the model's state dictionary
        torch.save(self._model.state_dict(), f"{file_path}/model_{iteration:04d}.pth")

        # If using a target network, save its state dict as well
        if self._use_target_net:
            torch.save(self._target_net.state_dict(), f"{file_path}/model_{iteration:04d}_target.pth")
        
    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        if torch.cuda.is_available():
            self._model.load_state_dict(torch.load("{}/model_{:04d}.pth".format(file_path, iteration)))
            if(self._use_target_net):
                self._target_net.load_state_dict(torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration)))
        else:
            model_state_dict = torch.load("{}/model_{:04d}.pth".format(file_path, iteration), map_location=torch.device('cpu'))
            self._model.load_state_dict(model_state_dict)
            if(self._use_target_net):
                target_net_state_dict = torch.load("{}/model_{:04d}_target.pth".format(file_path, iteration), map_location=torch.device('cpu'))
                self._target_net.load_state_dict(target_net_state_dict)



    def train_agent(self, batch_size=64, num_games=1, reward_clip=False):
        """
        Trains the agent using samples from the replay buffer.

        """
        # Sample from buffer
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)

        
        # Convert samples to PyTorch tensors and move them to device.
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        next_s = torch.tensor(next_s, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.int8).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).to(self.device)
        legal_moves = torch.tensor(legal_moves, dtype=torch.int8).to(self.device)

        if reward_clip:
            r = torch.sign(r)

        # training mode
        self._model.train()

        # use target network for prediction if enabled, else use main model.
        target_model = self._target_net if self._use_target_net else self._model

        # Q values for the next states, no gradient
        with torch.no_grad():  
            next_model_outputs = self._get_model_outputs(next_s, target_model)
        
        # discounted reward
        inf_tensor = torch.tensor(-np.inf).to(self.device)
        discounted_reward = r + self._gamma * torch.max(torch.where(legal_moves == 1, next_model_outputs, inf_tensor), dim=1).values.reshape(-1, 1) * (1 - done)

        # target Q-values
        target = self._get_model_outputs(s, target_model)
        target = (1 - a) * target + a * discounted_reward

        # predicted Q-values
        prediction = self._get_model_outputs(s)

        # loss between predicted Q-values and target Q-values.
        loss = nn.SmoothL1Loss()
        loss = loss(target, prediction)
        # Backpropagate the loss and update the model parameters.
        self._optimizer.zero_grad() 
        loss.backward()            
        self._optimizer.step()      

        return loss.item()

    def update_target_net(self):
        if self._use_target_net:
            # if using target network, update weights
            self._target_net.load_state_dict(self._model.state_dict())