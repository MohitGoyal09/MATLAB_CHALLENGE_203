classdef PropagateWithFlow < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal
%PropagateWithFlow Tracks a single rectangular ROI using Optical Flow.
%
%   This version uses the robust opticalFlowFarneback object.

    properties(Constant)
        % --- THE NAME HAS CHANGED ---
        Name = 'Propagate ROI with Flow (Robust)'; 
        Description = 'Tracks a single rectangular ROI using the Farneback optical flow method.';
        UserDirections = {...
            '1. Select or create a single Rectangle ROI label on the starting frame.', ...
            '2. Select the time range you want to automate.', ...
            '3. Click RUN to propagate the label across the time range.'};
    end

    properties
        OpticalFlow
        PreviousBBox
        LabelNameToAutomate
    end

    methods
        function isValid = checkLabelDefinition(~, labelDef)
            isValid = (labelDef.Type == labelType.Rectangle);
        end

        function isReady = checkSetup(~, labelsToAutomate)
            if height(labelsToAutomate) ~= 1
                errordlg('You must select exactly one rectangular ROI on the starting frame to track.', 'Incorrect Setup');
                isReady = false;
            else
                isReady = true;
            end
        end
    end

    methods
        function initialize(algObj, I, labelsToAutomate)
            % --- THIS IS THE FIX ---
            algObj.OpticalFlow = opticalFlowFarneback;
            
            algObj.LabelNameToAutomate = labelsToAutomate.Name{1};
            algObj.PreviousBBox        = labelsToAutomate.Position;
            
            firstGrayFrame = im2gray(I);
            algObj.OpticalFlow.estimateFlow(firstGrayFrame);
        end

        function autoLabels = run(algObj, I)
            currentGrayFrame = im2gray(I);
            
            % --- THIS IS THE FIX ---
            flow = algObj.OpticalFlow.estimateFlow(currentGrayFrame);
            
            newBBox = algObj.predictBBox(algObj.PreviousBBox, flow.Vx, flow.Vy);

            autoLabels = struct('Name', algObj.LabelNameToAutomate, 'Type', labelType.Rectangle, 'Position', newBBox);

            algObj.PreviousBBox  = newBBox;
        end
        function terminate(~)
        end
    end
    
    methods (Access = private)
        function newBBox = predictBBox(~, bbox, vx, vy)
            x1 = max(1, round(bbox(1))); y1 = max(1, round(bbox(2)));
            x2 = min(size(vx, 2), round(bbox(1) + bbox(3))); y2 = min(size(vx, 1), round(bbox(2) + bbox(4)));
            avg_vx = mean(vx(y1:y2, x1:x2), 'all'); avg_vy = mean(vy(y1:y2, x1:x2), 'all');
            new_x = bbox(1) + avg_vx; new_y = bbox(2) + avg_vy;
            newBBox = [new_x, new_y, bbox(3), bbox(4)];
        end
    end
end