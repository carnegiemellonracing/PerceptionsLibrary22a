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

from perc22a.predictors.utils.icp import icp 

# general imports
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=3)


class ConeState:
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
        self.icp_estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        self.icp_max_iters = max_iters
        self.icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.icp_max_iters)

        # for debugging
        self.timer = Timer()

        pass

    def _debug_correspondences(self, src_pos, target_pos, correspondences):

        # get positions without correspondences
        src_uncorr_mask = np.ones(src_pos.shape[0], dtype=bool)
        src_uncorr_mask[correspondences[:, 0]] = False
        target_uncorr_mask = np.ones(target_pos.shape[0], dtype=bool)
        target_uncorr_mask[correspondences[:, 1]] = False

        # plot the correspondences in colorful colors
        for i in range(len(correspondences)):
            src_idx, target_idx = correspondences[i,:]

            points = np.array([src_pos[src_idx], target_pos[target_idx]])
            plt.scatter(points[:, 0], points[:, 1])

        # plot points without correspondences in black
        for i in range(src_uncorr_mask.shape[0]):
            if src_uncorr_mask[i]:
                plt.scatter([src_pos[i,0]], [src_pos[i,1]], c="black")
        
        for i in range(target_uncorr_mask.shape[0]):
            if target_uncorr_mask[i]:
                plt.scatter([target_pos[i,0]], [target_pos[i,1]], c="black")

        plt.show()

        return

    def _cones_to_pc_arr(self, cones: Cones):
        blue, yellow, _ = cones.to_numpy()

        b0, b1 = np.zeros((blue.shape[0], 1)), np.ones((blue.shape[0], 1))
        y1 = np.ones((yellow.shape[0], 1))

        # x, y, z, yellow-count, total-count
        blue = np.concatenate([blue, b0, b1], axis=1)
        yellow = np.concatenate([yellow, y1, y1], axis=1)

        return np.concatenate([blue, yellow], axis=0)
    
    def _pc_arr_to_pc(self, pc_arr):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_arr[:, :3])
        return pcd
    
    def _icp_correspondence_o3d(self, prev_cone_pc_arr, curr_cone_pc_arr):
        # use ICP to align previous cone with target current cones

        # convert to open3d PointCloud objects
        prev_pcd = self._pc_arr_to_pc(prev_cone_pc_arr)
        curr_pcd = self._pc_arr_to_pc(curr_cone_pc_arr)

        # perform icp
        # TODO: using gps to inform init transformation would be good
        # might make it faster, but already lwk kinda fast

        self.timer.start("icp")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            prev_pcd, curr_pcd,
            self.icp_max_correspondence_dist,
            estimation_method=self.icp_estimation_method,
            criteria=self.icp_criteria
        )
        correspondences = np.asarray(reg_p2p.correspondence_set)
        self.timer.end("icp")

        # self._debug_correspondences(prev_cone_pc_arr, curr_cone_pc_arr, correspondences)
        return correspondences
    
    def _icp_correspondence_np(self, prev_cone_pc_arr, curr_cone_pc_arr):
        '''use own icp implementation that is singled threaded and pure NumPy'''
        
        # get source and target points (only concerned about x, y dimensions)
        src_points = prev_cone_pc_arr[:, :2]
        dest_points = curr_cone_pc_arr[:, :2]

        # perform icp
        # self.timer.start("icp")
        corr, T, corr_dists, iters = icp(
            src_points, dest_points,
            init_pose=None,
            max_iterations=self.icp_max_iters,
            max_corr_dist=self.icp_max_correspondence_dist
        )
        # self.timer.end("icp")

        # self._debug_correspondences(prev_cone_pc_arr, curr_cone_pc_arr, corr)
        return corr 
        
    
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
            self.cones_state_arr = self._cones_to_pc_arr(cones)
            return cones
        
        # save input cones for next iteration's prev_cones 
        new_cones = cones
        input_cones = cones.copy()

        # convert cones into a point cloud of cones
        new_cone_pc_arr = self._cones_to_pc_arr(new_cones)

        # perform icp to get correspondences
        corr = self._icp_correspondence_np(self.cones_state_arr, new_cone_pc_arr)  
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