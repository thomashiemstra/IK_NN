
close all;

scrsz = get(0,'ScreenSize');
fig1 = figure('Position',[scrsz(3)/4 scrsz(4)/5 scrsz(4)/1.5 scrsz(4)/1.5]);

h(1) = plot(x1,y1,'-x');
hold on;
h(2) = plot(x2,y2,'-o');
hold on;
h(3) = plot(x3,y3,'-d');
hold on;
h(4) = plot(x4,y4,'-s');
hold on;
h(5) = plot(x5,y5,'-^');
hold on;
h = legend([h(1) h(2) h(3) h(4) h(5) ],{'$50$','$100$','200','20\_20','20\_10'},'Interpreter','LaTex','fontsize',17);
hold off;
v = get(h,'title');
set(v,'string','Neurons:')
%ylim([0 0.03])
%xlim([0, 40000])

xlabel('$epochs$','Interpreter','LaTex','fontsize',17)
ylabel('$MSE$','Interpreter','LaTex','fontsize',17)