import numpy as np
#from scipy.signal import find_peaks



class TSHAPExplainer:

    def __init__(self, window_length = 20, stride = 5, interpolation = True, roi = True):
        self.window_length = window_length
        self.stride = stride
        self.interpolation = interpolation
        self.roi = roi
        
    def _get_window_mask(self, series_shape, window_channel, window_start, window_length):
        mask = np.zeros(series_shape)
        if window_start >= 0:
            mask[window_channel,window_start:(window_start + window_length)] = 1
        else:
            mask[window_channel,:(window_start + window_length)] = 1
        return mask

    def _explain_instance(self, fnc, input_sample, baselines):

        # all_samples = np.array([input_sample])
        n_channels, series_length = input_sample.shape        
        
        wd_pos = [(p,ws) for p in range(n_channels) for ws in range(0, min(series_length, series_length- self.window_length + self.stride), self.stride)]    
        wd_scores = self._calculate_window_attrib_scores(fnc, input_sample, baselines, wd_pos)
        

        reshaped_window_scores = np.zeros(input_sample.shape)
        for i in range(len(wd_pos)):
            ch, window_start = wd_pos[i]
            
            if self.interpolation and i > 0 and wd_pos[i][0] == wd_pos[i-1][0]: # not first window and same channel                 
                last_ws = wd_pos[i-1][1]
                reshaped_window_scores[ch, last_ws: window_start + 1] = np.linspace(wd_scores[i-1], wd_scores[i], self.stride + 1)
            else:
                reshaped_window_scores[ch, window_start] = wd_scores[i]


        window_exp = np.zeros(input_sample.shape)

        for c in range(n_channels):
            # for i in range(series_length - self.window_length + 1):
            for i in range(series_length):
                window_exp[c, i: i + self.window_length] += reshaped_window_scores[c, i] / (self.window_length * min(i+1,self.stride))
        
        return window_exp, reshaped_window_scores
    
    def _calculate_window_attrib_scores(self, fnc, input_sample, baselines, window_positions):
        all_samples = np.array([input_sample])
        payout_pos = [0 for _ in range(len(window_positions))]
        wd_scores = np.zeros(len(window_positions))
        bsize = len(baselines)        
        all_samples = np.vstack((all_samples, baselines))

        for i,ws in enumerate(window_positions):
            if len(ws) == 2:
                feature_mask = self._get_window_mask(input_sample.shape, ws[0], ws[1], self.window_length) # default window length
            else:
                feature_mask = self._get_window_mask(input_sample.shape, ws[0], ws[1], ws[2]) # custom window length
            payout_pos[i] = all_samples.shape[0]
            for baseline in baselines:            
                fused_samples = np.array([        
                input_sample * (1 - feature_mask) + baseline * feature_mask,
                input_sample * feature_mask + baseline * (1 - feature_mask),                    
                ])
                all_samples = np.vstack((all_samples, fused_samples))
      
        payout = fnc(all_samples)        

        for i, pp in enumerate(payout_pos):            
            val = 0
            for ii in range(bsize):
                val += ((payout[0] - payout[pp + ii*2]) + (payout[pp +ii*2+1] - payout[1 + ii]))/2
            val = val/bsize 
            wd_scores[i] = val
        
        return wd_scores


    def _explain_dataset(self, fnc, X, baselines):
        X_attribs = np.zeros(X.shape)
        X_roi_attribs = np.zeros(X.shape)
        for i in range(X.shape[0]):
            X_attribs[i] , window_scores = self._explain_instance(fnc, X[i], baselines)
            if self.roi:
                rois = self._find_rois_v2(window_scores)                   
                roi_attribs = self._calculate_window_attrib_scores(fnc, X[i], baselines, rois)                
                for roi,attrb in zip(rois, roi_attribs):
                    X_roi_attribs[i, roi[0], roi[1]: roi[1] + roi[2] + 1] = attrb / (roi[2] + 1)
           
        return X_attribs, X_roi_attribs
    

    def _find_rois_v2(self, wd_scores):

        abs_wd_scores = np.abs(wd_scores)
        rel_threshold = min(0.05*np.max(abs_wd_scores), np.percentile(abs_wd_scores,q=50))
        
        thresholded_scores =  np.zeros_like(wd_scores)
        thresholded_scores[wd_scores > rel_threshold] = 1
        thresholded_scores[wd_scores < -rel_threshold] = -1

        # rois = [[] for _ in range(thresholded_scores.shape[0])]
        rois = []
        for c in range(thresholded_scores.shape[0]):

            # group windows            
            candidate_start = 0 
            zones = []           
            while candidate_start < thresholded_scores.shape[1]:
                current_zone = thresholded_scores[c, candidate_start]
                candidate_end = candidate_start
                for ci in range(candidate_start + 1, thresholded_scores.shape[1]):
                    if thresholded_scores[c, ci] == current_zone:
                            candidate_end = ci
                    else:
                        break
                if current_zone != 0 or (candidate_end - candidate_start) > 3:
                    # if zones and current_zone == zones[-1][2]:
                    #     #print("Merging zones")
                    #     zones[-1][1] = candidate_end
                    # else:
                    zones.append([candidate_start, candidate_end, current_zone])
                # print(zones)
                candidate_start = candidate_end + 1
            
            zones.append([-1,-1,0]) # dummy zone

            for zi in range(len(zones)-1):                
                if zones[zi][2] != 0:                    
                    roi_start = zones[zi][0] + self.window_length//2 if \
                            zones[zi-1][2]  * zones[zi][2] < 0 else zones[zi][0] + self.window_length
                    roi_end = zones[zi][1] + self.window_length//2 if \
                            zones[zi+1][2] * zones[zi][2] < 0 else zones[zi][1] 
                    
                    if roi_end > roi_start:
                            #rois[c].append([roi_start, roi_end])
                        rois.append([c, roi_start, roi_end - roi_start])
                    
            
            # determine ROIs from zones
           

        return rois             

        
    
    def explain(self, X, baselines, model, clf_targets = None):

        if callable(model):
            predict_fnc = model
        elif hasattr(model, 'predict_proba') and callable(model.predict_proba): # classification
            if clf_targets is None:
                predict_fnc = lambda X: model.predict_proba(X)[:,1]
            else:
                target_index = np.argmax(model.classes_ == clf_targets)
                predict_fnc = lambda X: model.predict_proba(X)[:,target_index]
        else:            
            predict_fnc = model.predict # should return a scalar for regression

        window_exp, roi_exp = self._explain_dataset(predict_fnc, X, baselines)
     
        return window_exp, roi_exp
