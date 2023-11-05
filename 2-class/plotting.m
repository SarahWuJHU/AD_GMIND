%%
mean_accuracy = mean(valAcc, 3);
mean_auc = mean(valAUC, 3);
%%
mean_accuracy = mean_accuracy + mean(valAcc, 3);
mean_auc = mean_auc + mean(valAUC, 3);
%%
figure 
heatmap([0.1, 0.3, 0.6], [0.1, 0.3, 0.6], mean_accuracy, "title", "accuracy")

%%
figure
heatmap([0.1, 0.3, 0.6], [0.1, 0.3, 0.6], mean_auc, "title", "auc")

%%
Acc = zeros(3, 3, 10);
AUC = zeros(3, 3, 10);
%%
for i = 1:10
filename = "data_" + num2str(i) + "/setting/model" + num2str(i) + "/vis.mat"; 
load(filename)
[min_val, index] = min(valtotal_loss(:, :, 100:991), [], 3);
index = index + 99;
accuracy = zeros(3, 3);
auc = zeros(3, 3);
for a = 1:3
    for b = 1:3
        accuracy(a, b) = testAcc(a, b, index(a, b));
        auc(a, b) = testAUC(a, b, index(a, b));
    end
end
Acc(:, :, i) = accuracy;
AUC(:, :, i) = auc;
end

%%
acc = zeros(1, 10);
auc = zeros(1, 10);
f1 = zeros(1, 10);

%%
for i = 1:10
filename = "data_" + num2str(i) + "/setting/model" + num2str(i) + "/vis.mat"; 
load(filename)
[min_val, index] = min(valtotal_loss(100:591));
index = index + 99;
acc(i) = testAcc(index);
auc(i) = testAUC(index);
f1(i) = testF1(index);
end

%%
figure 
subplot(3, 1, 1)
hold on 
plot(acc, 'b')
plot(Accuracy, 'r')
plot(acc1)
plot(acc2)
legend("GMIND", "Random Forest", "Genetics", "Imaging")
title("Accuracy")
hold off

subplot(3, 1, 2)
hold on 
plot(auc, 'b')
plot(AUC, 'r')
plot(auc1)
plot(auc2)
legend("GMIND", "Random Forest", "Genetics", "Imaging")
title("AUC")
hold off

subplot(3, 1, 3)
hold on 
plot(f1, 'b')
plot(F1, 'r')
plot(f11)
plot(f12)
legend("GMIND", "Random Forest", "Genetics", "Imaging")
title("F1")
hold off
%%
accconf = reshape(Acc, [], 10);
aucconf = reshape(AUC, [], 10);
%%
figure
hold on
for i = 1:9
    plot(aucconf(i, :), 'b')
end
plot(f_acc, 'r')
title("AUC")
hold off
%%
figure 
hold on 
plot(mean(accconf), 'b')
plot(f_acc, 'r')

legend("GMIND", "Random Forest")
title("Accuracy")
hold off
%%
figure 
hold on 
plot(mean(aucconf), 'b')
plot(f_AUC, 'r')

legend("GMIND", "Random Forest")
title("AUC")
hold off