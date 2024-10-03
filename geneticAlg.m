% CW spec requires all code in one file

% Reads map as a binary image
map = im2bw(imread('random_map.bmp'));

% Fixed GA parameters
start = [1 1];
finish = [500 500];
mapSize = 500;

% Variable GA parameters
% (can be changed)
noOfPointsInSolution = 10;
mutationRate = 0.2;
populationSize = 1;
numGenerations = 1;

% Gets user input for selection, crossover, and mutation methods
selectionMethod = input('Enter the selection method (0 for RWS, 1 for Tournament, 2 for Rank-based): ');
crossoverMethod = input('Enter the crossover method (0 for Order Crossover, 1 for Position-Based Crossover): ');
mutationMethod = input('Enter the mutation method (0 for Flip Mutation, 1 for Exchange Mutation): ');

% Starts recording execution time
tic;

% Initialises the population
population = generatePopulation(populationSize, mapSize, noOfPointsInSolution);

% Main Genetic Algorithm Loop
for generation = 1:numGenerations
    % Evaluate the fitness of the population
    fitness = evaluateFitness(population, map);

    % Perform selection based on user input
    selectedIndices = selectIndividuals(selectionMethod, fitness, populationSize);

    % Initialises a new population
    newPopulation = zeros(populationSize, 2 * noOfPointsInSolution);

    % Performs crossover
    newPopulation = performCrossover(crossoverMethod, selectedIndices, population, newPopulation, noOfPointsInSolution, populationSize);

    % Performs mutation
    newPopulation = performMutation(mutationMethod, newPopulation, mutationRate, populationSize);

    % Replaces the old population with the new population
    population = newPopulation;
end

% Calculates and outputs execution time of GA
toc;

% Calculates the best path
solution = calcBestPath(population, map);

% Displays the best path
displayBestPath(solution, map, start, finish);

% Calculates and displays total Euclidean distance of the best path
totalDistance = calculateTotalDistance(path);
fprintf('Total Euclidean Distance: %.2f\n', totalDistance);

% Function to generate an initial population 
function population = generatePopulation(populationSize, mapSize, noOfPointsInSolution)
    % Create a matrix of (2*noOfPointsInSolution) random values between 0 and 1, 
    % and scales them to fit within the map size
    population = rand(populationSize, 2 * noOfPointsInSolution) .* (mapSize - 1) + 1;
end

% Function to evaluate the fitness of a population
function fitness = evaluateFitness(population, map)
    % Calculates the number of individuals (rows) in the population matrix
    numIndividuals = size(population, 1);
    % Initialises fitness column vector
    fitness = zeros(numIndividuals, 1);

    % Iterates over each individual in population
    for i = 1:numIndividuals
        % Extracts the path of the individual from population matrix
        path = population(i, :);
        % Calculates Euclidean distance between consecutive points in the path
        distances = calculateTotalDistance(path);
        % Calculates the penalty of the path
        penalty = calculatePenalty(path, map, distances);

        % Calculates the fitness of the individual
        % (Inverse of distances + penalty of path)
        fitness(i) =  1 / (sum(distances) + penalty);
    end
end

% Function to calculate the penalty for a path
function penalty = calculatePenalty(path, map, distances)
    % Checks if the path indices are valid and within the map scope
    validIndices = all(path > 0, 2) & path(:, 1) <= size(map, 1) & path(:, 2) <= size(map, 2);

    % Converts valid path indices to linear indices in the map
    obstacleIndices = sub2ind(size(map), round(path(validIndices, 1)), round(path(validIndices, 2)));

    % Checks if the path intersects with an obstacle (obstacle represented by 1 in the map)
    obstaclePositions = map(obstacleIndices) == 1;

    % Increments the penalty by a value proportional to the distance within the obstacle
    % (Penalty constant can be changed)
    penalty = sum(distances(validIndices) .* obstaclePositions) * 1000;
end

% Function to calulate the total distance of a path
function totalDistance = calculateTotalDistance(path)
    % Converts the path indices to coordinates
    x = path(1:2:end);
    y = path(2:2:end);

    % Calculates the Euclidean distance between consecutive points
    distances = sqrt(diff(x).^2 + diff(y).^2);

    % Sums up the distances to get the total distance
    totalDistance = sum(distances);
end

% Function to select individuals based on chosen method
function selectedIndices = selectIndividuals(selectionMethod, fitness, populationSize)
    % Initialises an array of zeros to store the selected indices
    selectedIndices = zeros(populationSize, 1);
   
    % Applies selection method
    switch selectionMethod
        case 0
            selectedIndices = RouletteWheelSelection(fitness);
        case 1
            selectedIndices = TournamentSelection(fitness, populationSize);
        case 2
            selectedIndices = RankBasedSelection(fitness, populationSize);
        otherwise
            error('Invalid selection method. Please enter 0, 1, or 2.');
    end
end

% Funcion to perform crossover based on chosen method
function newPopulation = performCrossover(crossoverMethod, selectedIndices, population, newPopulation, noOfPointsInSolution, populationSize)
    % Updates the population size based on the length of selectedIndices
    populationSize = length(selectedIndices);

    % Loops throught every other index
    for i = 1:2:populationSize
        % Checks the current index isn't the last
        if i + 1 <= populationSize

            % Gets the parents' indices
            parent1 = selectedIndices(i);
            parent2 = selectedIndices(i + 1);

            % Applies crossover to create two children
            switch crossoverMethod
                case 0
                    child1 = OrderCrossover(parent1, parent2, noOfPointsInSolution);
                    child2 = OrderCrossover(parent2, parent1, noOfPointsInSolution);
                case 1
                    child1 = PositionBasedCrossover(parent1, parent2);
                    child2 = PositionBasedCrossover(parent2, parent1);
                otherwise
                    error('Invalid crossover method. Please enter 0 or 1.');
            end

            % Adds the children to the new population
            newPopulation(i, :) = child1;
            newPopulation(i + 1, :) = child2;
        end
    end
end

% Function to perform mutation based on chosen method
function newPopulation = performMutation(mutationMethod, newPopulation, mutationRate, populationSize)
    % Iterates through the population
    for i = 1:populationSize
        % Checks if a random value is less than the mutation rate
        if rand() < mutationRate           
            % Applies mutation
            switch mutationMethod
                case 0
                    newPopulation(i, :) = FlipMutation(newPopulation(i, :));
                case 1
                    newPopulation(i, :) = ExchangedMutation(newPopulation(i, :));
                otherwise
                    error('Invalid mutation method. Please enter 0 or 1.');
            end
        end
    end
end

% Function to calculate the best path in the population
function solution = calcBestPath(population, map)

    % Evaluates the fitness of the final population
    finalFitness = evaluateFitness(population, map);

    % Finds index of individual with the largest fitness
    [~, bestIndex] = max(finalFitness);

    % Retrieves the path of the individual with the best fitness
    solution = population(bestIndex, :);
end

% Function to display the best path on the map
function path = displayBestPath(solution, map, start, finish)
    % Constructs the path variable 
    path = [start; [solution(1:2:end)' * size(map, 1), solution(2:2:end)' * size(map, 2)]; finish];

    % Clears current figure
    clf;
    % Displays binary map image
    imshow(map);

    % Draws a black rectangle around the map
    rectangle('position', [1 1 size(map) - 1], 'edgecolor', 'k');
    % Draws a line connecting the points in the path variable
    line(path(:, 2), path(:, 1));
end

% Function to perform Roulette Wheel Selection
function choice = RouletteWheelSelection(fitness)

    % Calculates the cumulative sum of fitness values
    % (probability of selection)
    accumulation = cumsum(fitness);

    % Generates random probability value
    p = rand();
    % Finds the index of the first element greater than the random probability
    chosen_index = find(accumulation > p, 1, 'first');

    % Returns the index of the chosen individual
    choice = chosen_index;
end

% Function to perform Tournament Selection
function choice = TournamentSelection(fitness, populationSize)
    
    % Randomly selects individuals from the population
    tournamentIndices = randperm(populationSize, populationSize);
    
    % Finds the index of the individual with the highest fitness in the tournament
    [~, winnerIndex] = max(fitness(tournamentIndices));
 
    % Returns the index of the individual
    choice = tournamentIndices(winnerIndex);
end

% Function to perform Rank-Based Selection 
function choice = RankBasedSelection(fitness, populationSize)

    % Sorts the fitness vector and gets the sorted indices
    [~, sortedIndices] = sort(fitness);
    % Initialises a vector of zeros 
    ranks = zeros(1, populationSize);
    % Assigns ranks to individuals based on their sorted indices
    ranks(sortedIndices) = 1:populationSize;
  
    % Calculates selection probabilities based on ranks
    selectionProbabilities = ranks / sum(ranks);
    
    % Calls RWS using selection probabilities
    choice = RouletteWheelSelection(selectionProbabilities);
end

% Function to perform Order Crossover
function offspring = OrderCrossover(parent1, parent2, noOfPointsInSolution)
    % Randomly selects starting index of substring
    substringStart = randi(noOfPointsInSolution/2);
    % Randomly selects ending index of the substring
    substringEnd = randi([substringStart + 1, noOfPointsInSolution*2]);
    % Extracts the selected substring from parent1
    substring = parent1(substringStart:substringEnd);

    % Initialises proto-child with the substring
    protoChild = zeros(1, length(parent1));
    protoChild(substringStart:substringEnd) = substring;

    % Determines the cities needed from parent2
    neededCities = setdiff(parent2, substring);

    % Places the needed cities into proto-child
    unusedIndices = find(protoChild == 0);
    protoChild(unusedIndices(1:length(neededCities))) = neededCities;

    % Returns the resulting offspring
    offspring = protoChild;
end

% Function to perform Position-Based Crossover
function offspring = PositionBasedCrossover(parent1, parent2)
    % Randomly selects a set of positions from parent1
    numPositions = randi(length(parent1));
    positions = randperm(length(parent1), numPositions);

    % Initialises proto-child with cities from parent1 at selected positions
    protoChild = zeros(1, length(parent1));

    % Uses matrix operations to set the values in proto-child
    protoChild(positions) = parent1(positions);

    % Determines the cities needed from parent2
    neededCities = setdiff(parent2, protoChild);

    % Places the needed cities into proto-child using matrix operations
    unusedIndices = find(protoChild == 0);
    protoChild(unusedIndices(1:length(neededCities))) = neededCities;

    offspring = protoChild;
end

% Function to perform exchanged mutation
function exchangedChromosome = ExchangedMutation(chromosome)
    % Randomly select two distinct positions
    positions = sort(randperm(length(chromosome), 2));

    % Use matrix operations to swap the values at the selected positions
    exchangedChromosome = chromosome;
    exchangedChromosome(positions) = exchangedChromosome(flip(positions));
end