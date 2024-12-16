function [state,options,optchanged] = trussOptHeuFigure(options, state, flag)
    persistent mass;
    optchanged = false;

    switch flag
        case 'init'
            hold on
            mass = [trussMass(state.Population(1, :)')];
        case 'iter'
            mass = [mass; trussMass(state.Population')];
        case 'done'
            plot(mass, 'LineWidth', 2);
            xlabel('Iterations / Generations');
            ylabel('Optimized Mass');
            hold off
            legend('GA');
        otherwise
    end
    
end