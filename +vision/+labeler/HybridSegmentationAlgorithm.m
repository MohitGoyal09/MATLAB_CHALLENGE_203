classdef HybridSegmentationAlgorithm < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal
% HybridSegmentationAlgorithm - The final, presentation-quality version.
% This algorithm uses a full pipeline: YOLO Detection -> Adaptive Threshold
% -> Morphological Cleaning -> Active Contour Refinement.

    properties(Constant)
        Name = 'Hybrid Segmentation (Presentation Quality)';
        Description = 'Produces high-quality, clean masks using a full refinement pipeline.';
        UserDirections = {'This is the final algorithm, producing the highest quality masks.', 'Press RUN to execute.'};
    end

    properties
        Detector;
        DetectionThreshold = 0.4;
    end

    methods
        function isValid = checkLabelDefinition(~, labelDef)
            isValid = (labelDef.Type == labelType.PixelLabel);
        end
        
        function isReady = checkSetup(algObj, ~)
            if any([algObj.ValidLabelDefinitions.Type] == labelType.PixelLabel)
                isReady = true;
            else
                errordlg('No PixelLabel definition found. Please create one.', 'Setup Error');
                isReady = false;
            end
        end
    end

    methods
        function initialize(algObj, ~, ~)
            disp('Initializing Presentation-Quality Hybrid Algorithm...');
            algObj.Detector = yolov4ObjectDetector('csp-darknet53-coco');
            disp('Initialization complete.');
        end

        function autoLabels = run(algObj, I)
            [bboxes, ~, labels] = detect(algObj.Detector, I, 'Threshold', algObj.DetectionThreshold);
            vehicleClasses = ["car", "truck", "bus"];
            vehicleBboxes = bboxes(ismember(string(labels), vehicleClasses), :);

            finalMask = false(size(I,1), size(I,2));
            
            for i = 1:size(vehicleBboxes, 1)
                bbox_int = floor(vehicleBboxes(i, :));
                [roi, roiRect] = imcrop(I, bbox_int);
                if isempty(roi) || size(roi,1) < 20 || size(roi,2) < 20, continue; end
                
                roiGray = im2gray(roi);
                
                % =============================================================
                % --- THE FINAL MASK CLEANING PIPELINE IS HERE ---
                % =============================================================
                
                % Step 2A: Get a rough initial mask.
                initialMask = imbinarize(roiGray, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
                
                % Step 2B: CLEANING - Keep only the largest object.
                cleanedMask = bwareafilt(initialMask, 1);
                
                % Step 2C: CLEANING - Fill any holes inside the object.
                cleanedMask = imfill(cleanedMask, 'holes');
                
                % Step 2D: CLEANING - Smooth the outline.
                se = strel('disk', 5);
                cleanedMask = imclose(cleanedMask, se);
                
                % Step 2E: FINAL REFINEMENT - Use active contours on the CLEAN mask.
                % This will produce a much better result.
                finalRoiMask = activecontour(roiGray, cleanedMask, 50, 'edge');
                
                % Place the final, refined mask back onto the main image.
                y_start = roiRect(2); x_start = roiRect(1);
                [roiHeight, roiWidth] = size(finalRoiMask);
                y_end = y_start + roiHeight - 1; x_end = x_start + roiWidth - 1;
                finalMask(y_start:y_end, x_start:x_end) = finalRoiMask;
            end
            
            pixelLabelDef = algObj.ValidLabelDefinitions([algObj.ValidLabelDefinitions.Type] == labelType.PixelLabel);
            pixelLabelDef = pixelLabelDef(1);
            targetLabelName = pixelLabelDef.Name;
            targetPixelID = pixelLabelDef.PixelLabelID;
            
            outputMask = zeros(size(I,1), size(I,2), 'uint8');
            outputMask(finalMask) = targetPixelID;
            
            autoLabels = categorical(outputMask, [0, targetPixelID], {'background', targetLabelName});
        end

        function terminate(~)
            disp('Hybrid segmentation complete.');
        end
    end
end