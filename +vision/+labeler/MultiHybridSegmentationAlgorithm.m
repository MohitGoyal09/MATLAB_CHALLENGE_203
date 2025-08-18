classdef MultiHybridSegmentationAlgorithm < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal
% HybridSegmentationAlgorithm - The final, multi-class, presentation-quality version.
% This algorithm detects multiple object classes (cars, people, etc.) and
% applies the corresponding PixelLabel for each, creating a multi-class segmentation.

    properties(Constant)
        Name = 'Multi-Class Hybrid Segmentation';
        Description = 'Detects and segments multiple classes (car, person, bicycle) with high-quality masks.';
        UserDirections = {...
            'CRITICAL SETUP: You MUST create a PixelLabel for each class you want to detect.', ...
            'The names must be exact: "car", "person", "bicycle", "truck", "bus", "motorbike".', ...
            'Press RUN to execute the algorithm.'};
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
                errordlg('No PixelLabel definitions found. Please create at least one.', 'Setup Error');
                isReady = false;
            end
        end
    end

    methods
        function initialize(algObj, ~, ~)
            disp('Initializing Multi-Class Hybrid Algorithm...');
            algObj.Detector = yolov4ObjectDetector('csp-darknet53-coco');
            disp('Initialization complete.');
        end

        function autoLabels = run(algObj, I)
            % Step 1: Detect all objects and get their class names.
            [bboxes, ~, classNames] = detect(algObj.Detector, I, 'Threshold', algObj.DetectionThreshold);
            
            % The list of classes we are interested in.
            supportedClasses = ["car", "truck", "bus", "person", "bicycle", "motorbike"];
            
            % Create an empty integer matrix to store the multi-class mask.
            labelMatrix = zeros(size(I,1), size(I,2), 'uint8');
            
            % Get all the PixelLabel definitions the user has created.
            allPixelLabelDefs = algObj.ValidLabelDefinitions([algObj.ValidLabelDefinitions.Type] == labelType.PixelLabel);

            % Step 2: Loop through each detected object.
            for i = 1:size(bboxes, 1)
                currentClassName = string(classNames(i));
                
                % Check if the detected class is one we support.
                if ~ismember(currentClassName, supportedClasses)
                    continue; % Skip if not a class of interest.
                end
                
                % --- DYNAMIC LABEL LOOKUP ---
                % Find the PixelLabel definition that matches the detected class name.
                isMatch = strcmp(currentClassName, {allPixelLabelDefs.Name});
                if ~any(isMatch)
                    disp(['Skipping detection: Please create a PixelLabel named "' char(currentClassName) '" to label it.']);
                    continue; % Skip if the user hasn't created a label for this class.
                end
                targetLabelDef = allPixelLabelDefs(isMatch);
                targetPixelID = targetLabelDef.PixelLabelID;
                % --- END LOOKUP ---
                
                % Now, perform the full segmentation pipeline for this object.
                bbox_int = floor(bboxes(i, :));
                [roi, roiRect] = imcrop(I, bbox_int);
                if isempty(roi) || size(roi,1) < 20 || size(roi,2) < 20, continue; end
                
                roiGray = im2gray(roi);
                
                initialMask = imbinarize(roiGray, 'adaptive', 'ForegroundPolarity', 'dark', 'Sensitivity', 0.4);
                cleanedMask = bwareafilt(initialMask, 1);
                cleanedMask = imfill(cleanedMask, 'holes');
                se = strel('disk', 5);
                cleanedMask = imclose(cleanedMask, se);
                finalRoiMask = activecontour(roiGray, cleanedMask, 50, 'edge');
                
                % "Paint" the final mask onto the label matrix using the correct ID.
                y_start = roiRect(2); x_start = roiRect(1);
                [roiHeight, roiWidth] = size(finalRoiMask);
                y_end = y_start + roiHeight - 1; x_end = x_start + roiWidth - 1;
                
                % Get the slice of the main label matrix.
                regionToUpdate = labelMatrix(y_start:y_end, x_start:x_end);
                % Only update the pixels belonging to the mask.
                regionToUpdate(finalRoiMask) = targetPixelID;
                % Place the updated slice back.
                labelMatrix(y_start:y_end, x_start:x_end) = regionToUpdate;
            end
            
            % Step 3: Convert the final integer matrix to the required 'categorical' format.
            allIDs = [0, allPixelLabelDefs.PixelLabelID];
            allNames = ["background", allPixelLabelDefs.Name];
            
            autoLabels = categorical(labelMatrix, allIDs, allNames);
        end

        function terminate(~)
            disp('Multi-class segmentation complete.');
        end
    end
end