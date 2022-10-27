%% cleaning
clear
close all
clc


%% Create training data set

x_train_points = transpose(linspace(0, 1, 1001));
y_train_points = sin(x_train_points*pi*2);

data = table(x_train_points, y_train_points); %create a table training data

%% Create NN 

layers =  [featureInputLayer(1)   % Input shape the network expects
          fullyConnectedLayer(16) % Dense layer of 16 neurons
          reluLaye                %activation funciton
          
          fullyConnectedLayer(16)
          reluLayer

          fullyConnectedLayer(1)
          tanhLayer

          regressionLayer % task - tells matlab to use MSE loss function
          ]
%% Training

%training options
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize',128, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(data, layers, options);

%% Testing

% create test dataset
x_test_points = transpose(linspace(0, 1, 852));
y_test_points = sin(x_test_points*pi*2);

% Get network to make predicitons on test x points
y_predictions = net.predict(x_test_points);


figure(1)
grid on
xlabel("x")
ylabel("y")
legend(["model predictions", "sin(2*pi*x)"])
plot(x_test_points, y_predictions)
hold on 
plot(x_test_points, y_test_points)
hold off
