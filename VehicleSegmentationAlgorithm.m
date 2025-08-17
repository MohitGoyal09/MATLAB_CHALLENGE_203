classdef VehicleSegmentationAlgorithm < vision.labeler.AutomationAlgorithm & vision.labeler.mixin.Temporal
% VehicleSegmentationAlgorithm - Automated vehicle detection and tracking.
% This is the final, robust version with stable tracking logic.

    properties(Constant)
        % --- Name updated to reflect the final version ---
        Name = 'Vehicle Tracker (Stable)';
        Description = 'Detects vehicles with YOLOv4 and uses a stable Kalman Filter for tracking.';
        UserDirections = {'This algorithm automatically detects and tracks vehicles.', 'Press RUN to execute the algorithm.'};
    end

    properties
        Detector
        ActiveTracks
        NextTrackID
    end

    methods
        function isValid = checkLabelDefinition(~, labelDef)
            isValid = (labelDef.Type == labelType.Rectangle);
        end
        
        function isReady = checkSetup(~, ~)
            isReady = true;
        end
    end

    methods
        function initialize(algObj, ~, ~)
            disp('Initializing YOLOv4 detector...');
            algObj.Detector = yolov4ObjectDetector('csp-darknet53-coco');
            algObj.ActiveTracks = struct(...
                'ID', {}, 'bbox', {}, 'kalmanFilter', {}, 'age', {}, ...
                'totalVisibleCount', {}, 'consecutiveInvisibleCount', {});
            algObj.NextTrackID = 1;
            disp('Initialization complete.');
        end

        function autoLabels = run(algObj, I)
            [bboxes, ~, labels] = detect(algObj.Detector, I, 'Threshold', 0.5);
            vehicleClasses = ["car", "truck", "bus"];
            detections = bboxes(ismember(string(labels), vehicleClasses), :);
            
            for i = 1:length(algObj.ActiveTracks)
                predictedBbox = predict(algObj.ActiveTracks(i).kalmanFilter);
                algObj.ActiveTracks(i).bbox = predictedBbox;
            end
            
            costMatrix = 1 - bboxOverlapRatio(vertcat(algObj.ActiveTracks.bbox), detections);
            [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(costMatrix, 0.7);
            
            for i = 1:size(assignments, 1)
                trackIdx = assignments(i, 1);
                detectionIdx = assignments(i, 2);
                correct(algObj.ActiveTracks(trackIdx).kalmanFilter, detections(detectionIdx, :));
                algObj.ActiveTracks(trackIdx).age = algObj.ActiveTracks(trackIdx).age + 1;
                algObj.ActiveTracks(trackIdx).totalVisibleCount = algObj.ActiveTracks(trackIdx).totalVisibleCount + 1;
                algObj.ActiveTracks(trackIdx).consecutiveInvisibleCount = 0;
            end
            
            for i = 1:length(unassignedTracks)
                trackIdx = unassignedTracks(i);
                algObj.ActiveTracks(trackIdx).age = algObj.ActiveTracks(trackIdx).age + 1;
                algObj.ActiveTracks(trackIdx).consecutiveInvisibleCount = algObj.ActiveTracks(trackIdx).consecutiveInvisibleCount + 1;
            end
            
            for i = 1:length(unassignedDetections)
                bbox = detections(unassignedDetections(i), :);
                kalmanFilter = configureKalmanFilter('ConstantVelocity', bbox, [200, 50], [100, 25], 100);
                newTrack = struct(...
                    'ID', algObj.NextTrackID, 'bbox', bbox, 'kalmanFilter', kalmanFilter, ...
                    'age', 1, 'totalVisibleCount', 1, 'consecutiveInvisibleCount', 0);
                algObj.ActiveTracks(end + 1) = newTrack;
                algObj.NextTrackID = algObj.NextTrackID + 1;
            end
            
            % =============================================================
            % --- THE FIXED, STABLE TRACKING LOGIC IS HERE ---
            % =============================================================
            if ~isempty(algObj.ActiveTracks)
                % A track is considered "lost" only if it has been invisible
                % for 15 consecutive frames. This makes the tracker patient.
                invisibleForTooLong = 15;
                lostIdx = [algObj.ActiveTracks.consecutiveInvisibleCount] >= invisibleForTooLong;
                algObj.ActiveTracks = algObj.ActiveTracks(~lostIdx);
            end

            autoLabels = [];
            if ~isempty(algObj.ActiveTracks)
                visibleTracks = algObj.ActiveTracks([algObj.ActiveTracks.consecutiveInvisibleCount] == 0);
                if ~isempty(visibleTracks)
                    bboxesToLabel = vertcat(visibleTracks.bbox);
                    autoLabels = struct('Name', 'car', 'Type', labelType.Rectangle, 'Position', bboxesToLabel);
                end
            end
        end

        function terminate(~)
            disp('Algorithm termination complete.');
        end
    end
end