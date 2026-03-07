function c = confusion_matrix(ws1, Xte, yte)

    label = zeros(284,1); % actual value
    for i = 1:284
        if (ws1(1:30)' * Xte(:,i)) + ws1(end) > 0 % positive class
            label(i) = 1;
        else % negative class
            label(i) = -1;
        end
    end
    
    confusion_matrix = [0 0; 0 0];
    for i = 1:284 % yte is predicted value
        if label(i) == 1 && yte(i) == 1 % a class
            confusion_matrix(1, 1) = confusion_matrix(1, 1) + 1;
        elseif label(i) == -1 && yte(i) == 1 % b class
            confusion_matrix(2, 1) = confusion_matrix(2, 1) + 1;
        elseif label(i) == 1 && yte(i) == -1 % c class
            confusion_matrix(1, 2) = confusion_matrix(1, 2) + 1;
        else % d class
            confusion_matrix(2, 2) = confusion_matrix(2, 2) + 1;
        end
    end
    
    disp(confusion_matrix)
    acc = sum(diag(confusion_matrix),'all') / sum(confusion_matrix,'all');
    fprintf("accuracy value is %f \n", acc)
end