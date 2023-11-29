 # Calculate size of flattened layer
                self.fc_input_size = self._calculate_flatten_size(config['board_size'], config['frames'])

                # Layers based on configuration
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=2, out_channels=conv1_config['filters'], kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(in_channels=conv1_config['filters'], out_channels=conv2_config['filters'], kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv2d(in_channels=conv2_config['filters'], out_channels=conv3_config['filters'], kernel_size=6, padding=1), nn.ReLU(),

                    # Dense layer
                    nn.Linear(self.fc_input_size, dense_config['units']), nn.ReLU(),  

                    # Output layer
                    nn.Linear(dense_config['units'], config['n_actions'])
                )

            def _calculate_flatten_size(self, board_size, frames):
                dummy_input = torch.zeros(1, frames, board_size, board_size)
                dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
                return int(torch.flatten(dummy_output, start_dim=1).size(1))




# Layers based on configuration
                self.conv1 = nn.Conv2d(10,64,2,10)
                self.conv2 = nn.Conv2d(64,64,1,1)
                self.conv3 = nn.Conv2d(64,64,1,1)