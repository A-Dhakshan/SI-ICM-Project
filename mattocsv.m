%for i = 0:24
%importfile1(sprintf("sim%d.mat",i))
%writematrix(stacks_discrete,sprintf('sim%d.csv',i))
%end

importfile1(sprintf("sim8.mat"))
writematrix(stacks_discrete,sprintf('sim8.csv'))

function importfile1(fileToRead1)
%IMPORTFILE1(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%  Auto-generated by MATLAB on 01-Oct-2024 16:46:05

% Import the file
newData1 = load('-mat', fileToRead1);

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end
end
