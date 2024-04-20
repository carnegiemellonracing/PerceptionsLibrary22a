'''ConeState.py

This file contains a class for using prior color estimates of cones to inform
and recolor new/incoming cone color estimates. The current implementation
uses ICP to determine correspondences between cones that have been seen (in state)
and new incoming cones. 

Then, each cone in the state has a counter for the
number of timesteps/prediction iterations they have been in the state
and another counter for the number of times the lidar coloring part of
the pipeline predicted them as yellow. Then, for cones that obtain
correspondences, their counts are updated based on the pipeline's color
predictions at that iteration. Cones in state that are not seen in the new
set of cones are discarded, and cones that haven't found an associated cone
in the existing state are added as a new cone to the state.
'''

# perc22a imports
from perc22a.predictors.utils.cones import Cones
from perc22a.utils.Timer import Timer

import perc22a.predictors.utils.icp as icp

# general imports
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)


class ConeState:
    '''Maintains a state of the cones that are being produced by various
    pipelines. In order to maintain additional metadata, the cones are 
    represented as a numpy array in the `self.cones_state_arr` attribute
    
    The structure of a state array for a cone is described as follows.
    
    The state array representation is going to be as follows (N x 7) array
        The 7 columns are defined as follows
            0. x position of cone
            1. y position of cone
            2. z position of cone
            3. # times cone has been seen as a yellow cone
            4. # times cone has been received by .update() call
            5. # times cone has not been seen (after being initially seen)
            6. # times cone has not been consecutively seen
    '''


    # TODO: determine what is the best merging policy?
    #   1. do it like counting and take the color with the highest count
    #   2. do we assume that the existing state is correct and ignore the old state
    #       in this case, what do if error in new incoming cone?

    # TODO: could accumulate transformations to get more cones in state
    # TODO: maybe just use ICP to inform decision about seed for lidar coloring?

    # TODO: weaknesses
    #   1. if a cone disappears, it's counts get totally refreshed which isn't good
    #   2. if looking at side, then new cones coming in might propogate wrong color
    #   - need uncorrelated cones to at least get colors from corresponded colors

    def __init__(self, max_correspondence_dist=1.25, max_iters=30):
        self.cones_state_arr = None

        # search correspondences over 1m radius
        self.icp_max_correspondence_dist = max_correspondence_dist
        self.icp_max_iters = max_iters 

        # for debugging
        self.timer = Timer()

        pass

    def _cones_to_state_arr(self, cones: Cones):
        '''Converts a new cones object to a state array representation'''
        blue, yellow, _ = cones.to_numpy()

        b0, b1 = np.zeros((blue.shape[0], 1)), np.ones((blue.shape[0], 1))
        y1 = np.ones((yellow.shape[0], 1))

        # x, y, z, yellow-count, seen-count, missed-count, consecutively-missed-count
        blue = np.concatenate([blue, b0, b1], axis=1)
        yellow = np.concatenate([yellow, y1, y1], axis=1)

        return np.concatenate([blue, yellow], axis=0)
       
    def _icp_transform_and_corr(self, src_points, dest_points):
        '''use own icp implementation that is singled threaded and pure NumPy

        This function uses ICP to determine the transformation between two
        sets of cones from their state arrays and also determines the 
        correspondences between the two

        Arguments:
            - prev_cone_state_arr: state array of cones from prior timesteps
                these cones have positions w.r.t. car at prior timestep
            - curr_cone_state_arr: state array of cones from current timestep
        
        Returns:
            - transformed_prev_state: prior state of cones but with x and y
                positions to be updated w.r.t. car at current timestep using ICP
            - corr: (K x 2) integer nparray of indices representing
                corresponences between prev and curr cone state arrays
                note: (K <= min(# num prev cones, # num curr cones))
        ''' 

        # perform icp
        corr, T, corr_dists, iters, transformed_src = icp.icp(
            src_points, dest_points,
            init_pose=None,
            max_iterations=self.icp_max_iters,
            max_corr_dist=self.icp_max_correspondence_dist
        )

        # icp.debug_correspondences(src_points, dest_points, corr)
        
        # update the old positions of the cones using the transformation
        # useful for updating position of uncorrelated cones
        return transformed_src, corr
        
    
    def _update_state_prob(self, cones_state_arr, new_cone_arr, correspondences):

        # NOTE: this is where the primary update policy is implemented
        # and how merging past estimates works with merging current estimates

        # 3 groups
        # in correspondence set (update positions and update counts)
        # cone in cone state not in correspondence set (remove from state)
        # cone in new cones (add into state, initialize counts)

        state_corr = correspondences[:, 0]
        new_corr = correspondences[:, 1]

        # 1. update cones found in the correspondence set
        corr_state_cones = cones_state_arr[state_corr, :]
        corr_new_cones = new_cone_arr[new_corr, :]

        # for each corr new cone, update counts with correlated state cones
        corr_new_cones[:, 3] += corr_state_cones[:, 3]
        corr_new_cones[:, 4] += corr_state_cones[:, 4]

        # 2. add new cones not found in the correspondence set
        new_uncorr_mask = np.ones(new_cone_arr.shape[0], dtype=bool)
        new_uncorr_mask[new_corr] = False

        uncorr_new_cones = new_cone_arr[new_uncorr_mask, :]

        # now merge correlated and uncorrelated new cones for updated state
        new_state_arr = np.concatenate([corr_new_cones, uncorr_new_cones], axis=0)
        return new_state_arr


    def _state_to_cones_prob(self, cones_state_arr):

        # get indices for blue and yellow cones based on predictions
        yellow_prob = cones_state_arr[:, 3] / cones_state_arr[:, 4]
        blue_idxs = np.where(yellow_prob < 0.5)
        yellow_idxs = np.where(yellow_prob >= 0.5)

        blue_cones_arr = cones_state_arr[blue_idxs][:, :3]
        yellow_cones_arr = cones_state_arr[yellow_idxs][:, :3]
        orange_cones_arr = np.zeros((0, 3))

        return Cones.from_numpy(blue_cones_arr, yellow_cones_arr, orange_cones_arr)
    

    def update(self, cones: Cones):
        if len(cones) == 0:
            # TODO: is this the best behavior, should use prior state if possible?
            return cones

        if self.cones_state_arr is None or self.cones_state_arr.shape[0] <= 1:
            self.prev_cones = cones
            self.cones_state_arr = self._cones_to_state_arr(cones)
            return cones
        
        # save input cones for next iteration's prev_cones 
        new_cones = cones
        input_cones = cones.copy()

        # convert cones into a point cloud of cones
        new_cone_pc_arr = self._cones_to_state_arr(new_cones)

        # use icp to get correspondences and set cone state w.r.t curr car pos
        self.cones_state_arr[:, :2], corr = self._icp_transform_and_corr(
            self.cones_state_arr[:, :2], 
            new_cone_pc_arr[:, :2]
        )  
        if corr is None:
            # if unable to find correspondences
            # return current cones without integrating into state
            # TODO: must determine if this is the appropriate behavior
            # typically occurs when many cones disappear or no cones available
            return cones

        # create some new cones and update prior cone state
        self.prev_cones = input_cones
        self.cones_state_arr = self._update_state_prob(
            self.cones_state_arr,
            new_cone_pc_arr,
            corr
        )

        # convert existing state into a Cones object
        # print(np.round(self.cones_state_arr, 3))
        cones = self._state_to_cones_prob(self.cones_state_arr)

        return cones
    
    def get_svm_cones(self):
        return None