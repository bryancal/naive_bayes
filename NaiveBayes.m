%{
    Multinomial Naive Bayesian Classifier
    Support to be added for standard NaiveBayes, Kernels, and to offer greater control over smoothing.
    Author: Bryan Callaway
%}
classdef NaiveBayes < handle
    properties
        laplaceSmoothing = 1;
        laplaceConstant = 1;
        classes;
        features;
        phiY = [];
        phiK_YC = [];
        numTrainDocs;
    end
    
    methods
        function obj = NaiveBayes(self)
            % Null constructor.
            % Add processing for classification categories.
        end
        
        function self = train(self, class, trainMatrix, varargin)
            inputs = inputParser;
            addRequired('class');
            addRequired('trainMatrix', @isnumeric);
            addOptional('lpSmoothing', self.laplaceSmoothing, @isnumeric);
            addOptional('laplaceConstant', self.laplaceConstant, @isnumeric);
            parse(inputs, varargin{:});
            
            self.classes = unique(class, 'sorted');
            self.numTrainDocs = size(trainMatrix, 1);
            self.features = size(trainMatrix, 2);
            laplaceDenom = lpSmoothing * self.features; 
            % Initialize  & compute training parameters.
            self.phiK_YC = zeros(self.features, self.classes);
                        
            % Estmate class conditional probabilities.
            for i = 1:len(self.classes)
                self.phiY(i) = sum(class == self.classes(i)) / self.numTrainDocs;
                self.phiK_YC(:, i) = ((class == self.classes(i - 1)) * trainMatrix + laplaceConstant) ./...
                        ((class == (i - 1)) * sum(trainMatrix, 2) + laplaceDenom);
            end
        end
        
        function [predictions, errorRate] = classify(self, class, trainMatrix)
            trainingSize = size(trainMatrix, 1);
            logProbs = zeros(trainingSize, length(self.classes));
            for i = 1:trainingSize
                for j = 1:self.classes
                  logProbs(i, j) = sum(log(self.phiK_YC(find(trainMatrix(i, :)), j)) .* nonzeros(trainMatrix(i, :))) + log(self.phiY(j));
                end
            end
            
            % Find max column in each row; idx is assigned class.
            [~, I] = max(logProbs, [], 2);
            
            predictions = I - 1; % rescale index.
            % Compute the error rate.
            errorRate = sum(abs(predictions - class') > 0) / trainingSize;
        end
    end
end
