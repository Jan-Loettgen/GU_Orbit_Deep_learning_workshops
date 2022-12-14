%% cleaning
clear
close all
clc


%% Create training data set

x_train_points = transpose(linspace(0, 1, 2001));

shuffled_x = zeros(2001, 1);

i = 1;
for index = randperm(2001)
    shuffled_x(i, 1) = x_train_points(index, 1);
    i=i+1;
end

y_train_points = sin(shuffled_x*pi*2);

data = table(shuffled_x, y_train_points); %create a table training data

%% Create NN 

layers =  [featureInputLayer(1)   % Input shape the network expects
          fullyConnectedLayer(32) % Dense layer of 16 neurons
          reluLayer                %activation funciton
          
          fullyConnectedLayer(32)
          reluLayer

          fullyConnectedLayer(1)
          tanhLayer

          regressionLayer % task - tells matlab to use MSE loss function
          ]
%% Training

%training options
options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
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
plot(x_test_points, y_predictions)
hold on 
plot(x_test_points, y_test_points)
legend(["model predictions", "sin(2*pi*x)"])
hold off
