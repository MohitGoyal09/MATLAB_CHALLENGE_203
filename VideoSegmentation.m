classdef VideoSegmentation < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal
% VideoSegmentationAlgorithm - Advanced video object segmentation.
% This version uses a robust Blob Analysis method to find whole objects.

    properties(Constant)
        Name = 'Blob Analysis Segmentation';
        Description = 'Finds object-like blobs using edge closing and filtering.';
        UserDirections = {...
            'This algorithm is better at finding whole objects.', ...
            '1. Click Settings. Adjust the "Edge Closing Radius" to connect broken edges on cars.', ...
            '2. Adjust "Min Object Size" to filter out road markings and noise.', ...
            '3. Run the algorithm and review the results.'};
    end

    properties
        EdgeThreshold = 0.04; 
        MinObjectSize = 2000; % Increased default size
        % --- NEW PARAMETER ---
        ClosingRadius = 5;    % Radius for connecting broken edges
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
        
        function settingsDialog(algObj)
            prompt = {'Edge Threshold (lower is more sensitive):', ...
                      'Minimum Object Size (pixels):', ...
                      'Edge Closing Radius (pixels):'}; % <-- New setting
            dlgtitle = 'Blob Analysis Settings';
            dims = [1 60];
            definput = {num2str(algObj.EdgeThreshold), num2str(algObj.MinObjectSize), num2str(algObj.ClosingRadius)};
            
            answer = inputdlg(prompt, dlgtitle, dims, definput);
            
            if ~isempty(answer)
                algObj.EdgeThreshold = str2double(answer{1});
                algObj.MinObjectSize = str2double(answer{2});
                algObj.ClosingRadius = str2double(answer{3}); % <-- New setting
                disp('Settings updated.');
            else
                disp('Settings dialog cancelled.');
            end
        end
    end

    methods
        function initialize(~, ~, ~)
            disp('Initializing Blob Analysis Algorithm...');
        end

        % =================================================================
        % --- THE NEW, SMARTER RUN METHOD IS HERE ---
        % =================================================================
        function autoLabels = run(algObj, I)
            if size(I, 3) == 3, currentGray = rgb2gray(I); else, currentGray = I; end
            
            % Step 1: Find strong edges.
            edges = edge(currentGray, 'Canny', algObj.EdgeThreshold);
            
            % Step 2: Connect nearby edges to form closed shapes (blobs).
            % This is the key step to group the car edges together.
            se = strel('disk', algObj.ClosingRadius);
            closedEdges = imclose(edges, se);
            
            % Step 3: Fill the interior of the blobs to make them solid.
            filledBlobs = imfill(closedEdges, 'holes');
            
            % Step 4: Remove small blobs that are likely noise or road markings.
            finalMask = bwareaopen(filledBlobs, algObj.MinObjectSize);
            
            % Step 5: Get the name and ID of the PixelLabel defined by the user.
            pixelLabelDef = algObj.ValidLabelDefinitions([algObj.ValidLabelDefinitions.Type] == labelType.PixelLabel);
            pixelLabelDef = pixelLabelDef(1);
            targetLabelName = pixelLabelDef.Name;
            targetPixelID = pixelLabelDef.PixelLabelID;
            
            % Step 6: Prepare the final output mask.
            outputMask = zeros(size(I,1), size(I,2), 'uint8');
            outputMask(finalMask) = targetPixelID; % Paint the final blobs
            
            % Step 7: Convert the matrix to the required 'categorical' format.
            autoLabels = categorical(outputMask, [0, targetPixelID], {'background', targetLabelName});
        end

        function terminate(~)
            disp('Algorithm termination complete.');
        end
    end
end