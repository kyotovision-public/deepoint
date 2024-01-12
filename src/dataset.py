from dataclasses import dataclass
import csv
import cv2
import yaml
import torch
from torch.utils.data import Dataset
import torchvision
from pathlib import Path
import pickle
import numpy as np
import yaml
from PIL import Image
from einops import rearrange

import logging


def pad_sequence(
    sequences: list[torch.Tensor], pad_tensor: torch.Tensor
) -> torch.Tensor:
    num_trailing_dims = len(sequences[0].size()[1:])
    out_dims = (len(sequences), 1) + (1,) * num_trailing_dims
    out_tensor = pad_tensor.repeat((out_dims))
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor
    return out_tensor


def is_l_or_r(txt: str) -> str:
    if "r" in txt:
        return "r"
    elif "l" in txt:
        return "l"
    else:
        return "?"


@dataclass
class DPDataFrame:
    venue_path: str
    session: str
    camera_id: int
    frames: list[int]
    action_id: int
    detection_list: list[dict]
    lr: str
    has_uncertain_frame: bool


class DPDataset(Dataset):
    EMPTY_DETECTION = {
        "keypoints": [0 for _ in range(17 * 3)],
        "bbox": [0 for _ in range(4)],
        "score": 0,
        "category_id": 1,
    }
    CATEGORY2ID: dict[str, int] = {
        "pad": 0,
        "cls": 1,
        "nose": 2,
        "left_eye": 3,
        "right_eye": 4,
        "left_ear": 5,
        "right_ear": 6,
        "left_shoulder": 7,
        "right_shoulder": 8,
        "left_elbow": 9,
        "right_elbow": 10,
        "left_wrist": 11,
        "right_wrist": 12,
        "left_hip": 13,
        "right_hip": 14,
        "left_knee": 15,
        "right_knee": 16,
        "left_ankle": 17,
        "right_ankle": 18,
    }
    JOINT_THRESH: float = 0.4
    SCORE_THRESH: float = 0.4
    NUM_JOINTS: int = 17

    def usable_frames_in_venue(
        self, venue_path, label_dir, pifpaf_detection_dict
    ) -> list[DPDataFrame]:
        data: dict[tuple[str, int, int], DPDataFrame] = dict()
        with (Path(label_dir) / "timeinfo.yaml").open("r") as f:
            timeinfo = yaml.safe_load(f)
            filt = timeinfo["filt"]
            sessions = timeinfo["sessions"]
            del timeinfo
        for session in sessions:
            session: str
            logging.debug(f"{session}")
            for camera_id in range(15):
                with (Path(label_dir) / (session + ".txt")).open("r") as f:
                    # The first line of txt file tells you the range of movie frames that can be used
                    reader = csv.reader(f)
                    row = reader.__next__()
                    valid_frame_start, valid_frame_end = (
                        int(row[0]) // filt,
                        int(row[1]) // filt,
                    )
                    negative_frames = set(
                        range(
                            int(valid_frame_start) + (self.cfg.model.tlength - 1),
                            int(valid_frame_end),
                        )
                    )
                    for row in reader:
                        # From the second line, each line shows each action's target marker id, starting/end time and hand (`l` or `r`).
                        # The line that starts with `#` is comment.
                        try:
                            if row[0].startswith("#"):
                                continue
                            action_id, start_frame, end_frame = (
                                int(row[0]),
                                int(row[1]) // filt,
                                int(row[2]) // filt,
                            )
                        except IndexError:
                            continue
                        except ValueError:
                            logging.warning(f"warning: ValueError detected at {row}")
                            break
                        try:
                            lr = is_l_or_r(row[3])
                        except IndexError:
                            lr = "?"
                        if action_id not in self.rev_action_class_dict.keys():
                            continue
                        for frame in range(
                            max(
                                valid_frame_start + (self.cfg.model.tlength - 1),
                                start_frame,
                            ),
                            min(end_frame, valid_frame_end),
                        ):
                            # `images[frame-(token_length-1), ..., frame-1, frame]` are the input
                            # Given these, we estimate the pointing actions at `frame`
                            # If all of pose keypoint estimation in the frame is higher than the self.SCORE_THRESH, add this to the dataset
                            has_uncertain_frame = False
                            frames = []
                            detection_list = list()
                            for i in reversed(range(self.cfg.model.tlength)):
                                detection = pifpaf_detection_dict[session][camera_id][
                                    frame - i
                                ]
                                frames.append(frame - i)
                                # `detection` is the list of humans in the image. The one with a highest certainty comes to the first.
                                if (
                                    len(detection) == 0
                                    or detection[0]["score"] < self.SCORE_THRESH
                                ):
                                    has_uncertain_frame = True
                                try:
                                    detection_list.append(
                                        max(detection, key=lambda x: x["bbox"][3])
                                    )
                                except:
                                    detection_list.append(self.EMPTY_DETECTION)
                            if has_uncertain_frame:
                                continue
                            data[session, camera_id, frame] = DPDataFrame(
                                venue_path,  # ex:17-living
                                session,  # ex: take1
                                camera_id,  # ex: 0
                                frames,  # ex: 183
                                action_id,
                                detection_list,
                                lr,
                                has_uncertain_frame,
                            )
                            negative_frames -= {frame}

                    if self.NON_POINTING_ACTIONID in self.rev_action_class_dict.keys():
                        for frame in negative_frames:
                            if not (
                                valid_frame_start
                                < frame
                                < valid_frame_end - self.cfg.model.tlength
                            ):
                                continue

                            # If all of pose keypoint estimation in the frame is higher than the self.SCORE_THRESH, add this to the dataset
                            has_uncertain_frame = False
                            frames = []
                            detection_list = list()
                            for i in reversed(range(self.cfg.model.tlength)):
                                try:
                                    detection = pifpaf_detection_dict[session][
                                        camera_id
                                    ][frame - i]
                                except:
                                    detection = []
                                frames.append(frame - i)
                                if (
                                    len(detection) == 0
                                    or detection[0]["score"] < self.SCORE_THRESH
                                ):
                                    has_uncertain_frame = True
                                try:
                                    detection_list.append(
                                        max(detection, key=lambda x: x["bbox"][3])
                                    )
                                except:
                                    detection_list.append(self.EMPTY_DETECTION)
                            if has_uncertain_frame:
                                continue
                            data[session, camera_id, frame] = DPDataFrame(
                                venue_path,
                                session,
                                camera_id,
                                frames,
                                self.NON_POINTING_ACTIONID,
                                detection_list,
                                "?",
                                has_uncertain_frame,
                            )

        return list(map(lambda x: x[1], sorted(data.items())))

    def __init__(
        self,
        keypoints_path: Path,
        cfg,
        omit_image=False,
    ):
        """
        omit_image: If given, doesn't return image or featvec
        """

        self.cfg = cfg
        self.omit_image = omit_image

        self.action_class_dict = {0: [-1], 1: list(range(0, 100))}
        self.num_classes = len(self.action_class_dict)
        with (keypoints_path / "triangulation.pickle").open("rb") as f:
            self.coords = pickle.load(f)
        with (keypoints_path / "collected_json.pickle").open("rb") as f:
            self.collected_pifpaf_json = pickle.load(f)

        self.NON_POINTING_ACTIONID = -1

        self.rev_action_class_dict = {
            v: key for (key, val) in self.action_class_dict.items() for v in val
        }

        logging.info("Initializing Dataset...")

        self.datalist: list[DPDataFrame] = list()

        self.markerdict = dict()
        self.jointcoords_dict = dict()
        self.cameraparam_dict = dict()

        # for each venues, Process annotations
        for venue_name in cfg.data.venue_names:
            venue_path, label_dir = (
                cfg.data.data_root + venue_name,
                cfg.data.label_root + venue_name,
            )
            joint_coords = self.coords[Path(venue_path).name]
            pifpaf_annotation = self.collected_pifpaf_json[Path(venue_path).name]
            logging.info(f"Loading {venue_path=}")
            self.datalist.extend(
                self.usable_frames_in_venue(venue_path, label_dir, pifpaf_annotation)
            )
            marker_coord_path = Path(venue_path) / "venue-info" / "marker_corners.npz"
            assert marker_coord_path.exists(), f"{marker_coord_path} not found."
            with np.load(marker_coord_path) as data:
                self.markerdict[venue_path] = {
                    "valid_ids": data["valid_ids"],
                    "corners": data["corners"],
                }
            camera_param_path = Path(venue_path) / "venue-info" / "params.pickle"
            with camera_param_path.open("rb") as f:
                self.cameraparam_dict[venue_path] = pickle.load(f)
            self.jointcoords_dict[venue_path] = joint_coords

        logging.info(f"Initialized Dataset. Frames in dataset:{len(self)}.")

    def __getitem__(self, idx: int):
        data = self.datalist[idx]
        venue_path = data.venue_path
        session = data.session
        camera_id = data.camera_id
        frames = data.frames
        action_id = data.action_id
        detection_list = data.detection_list
        lr = data.lr
        has_uncertain_frame = data.has_uncertain_frame

        # load featvec or images
        featvec_joints, featvec_bbs, featvec_imgs = [], [], []
        images = []
        if self.omit_image:
            pass
        else:
            if self.cfg.use_preprocessed:
                for f in frames:
                    image_path = str(
                        Path(venue_path)
                        / session
                        / f"{camera_id:02d}"
                        / f"{f+1:010d}.jpg"  # The output of ffmpeg is 1-indexed
                    )
                    with Path(
                        #  f"{image_path.replace('frames','featvec-result')}.pickle"
                        f"{image_path.replace('frames','featvec-result-scratch')}.pickle"  # スクラッチ領域へのリンク
                    ).open("rb") as f:
                        featvec = pickle.load(f)
                        featvec_joint, featvec_bb, featvec_img = featvec
                        if self.cfg.model.featvec_joint:
                            featvec_joints.append(featvec_joint)
                        if self.cfg.model.featvec_bb:
                            featvec_bbs.append(featvec_bb)
                        if self.cfg.model.featvec_img:
                            featvec_imgs.append(featvec_img)
                featvec_joints = torch.tensor(np.array(featvec_joints))
                featvec_bbs = torch.tensor(np.array(featvec_bbs))
                featvec_imgs = torch.tensor(np.array(featvec_imgs))
            else:
                for f in frames:
                    image_path = str(
                        Path(venue_path)
                        / session
                        / f"{camera_id:02d}"
                        / f"{f+1:010d}.jpg"  # The output of ffmpeg is 1-indexed
                    )
                    image = torchvision.io.read_image(image_path)
                    images.append(image.float() / 255)
                images = torch.stack(images)

        # keypoints
        if has_uncertain_frame:
            keypoints = torch.zeros((self.cfg.model.tlength, 17, 3))
            width = torch.ones(self.cfg.model.tlength)
            height = torch.ones(self.cfg.model.tlength)
            bbox = torch.zeros((self.cfg.model.tlength, 4))
        else:
            keypoints = rearrange(
                torch.tensor([d["keypoints"] for d in detection_list]),
                f"token_length  (num_joints xyp) -> token_length num_joints xyp",
                xyp=3,
            )
            bbox = torch.tensor([d["bbox"] for d in detection_list])
            width, height = bbox[:, 2], bbox[:, 3]

        # joint_id
        joint_id = [
            torch.where(keypoints[i, :, 2] > self.JOINT_THRESH)[0]
            for i in range(self.cfg.model.tlength)
        ]
        if self.cfg.filter_joint is not None:
            if self.cfg.filter_joint == "hand":
                used_joint_name = {"left_wrist", "right_wrist"}
            elif "handhead":
                used_joint_name = {
                    "left_wrist",
                    "right_wrist",
                    "nose",
                    "left_eye",
                    "right_eye",
                    "left_ear",
                    "right_ear",
                }
            else:
                raise NotImplementedError
            used_joint_id = torch.tensor(
                list(
                    map(
                        lambda key: self.CATEGORY2ID[key],
                        used_joint_name,
                    )
                )
            )
        else:
            # 全部OK
            used_joint_id = torch.tensor(list(self.CATEGORY2ID.values()))
        joint_id = [
            torch.where(torch.isin(j, used_joint_id), j, self.CATEGORY2ID["pad"])
            for j in joint_id
        ]
        # For simplicity, use the last token as the joint special token
        pad_jointid_tensor = torch.full((self.NUM_JOINTS + 1,), self.CATEGORY2ID["pad"])
        pad_jointid_tensor[-1] = self.CATEGORY2ID["cls"]
        joint_id = pad_sequence(joint_id, pad_tensor=pad_jointid_tensor)

        # joint position tokens
        abs_joint_pos = keypoints[:, :, :2]
        chest_coord = (keypoints[:, 5, :2] + keypoints[:, 6, :2])[:, None, :] / 2
        # Both are supposed to be in range of [-1, 1]
        rel_joint_pos = (abs_joint_pos - chest_coord) / torch.stack(
            [width, height], dim=-1
        )[:, None, :]

        # Add special token to joint position tokens
        rel_joint_pos = torch.concat(
            (rel_joint_pos, torch.tensor([0]).repeat(self.cfg.model.tlength, 1, 2)),
            dim=1,
        )

        # src_key_padding_mask for transformer
        src_key_padding_mask = joint_id == self.CATEGORY2ID["pad"]

        # answer
        try:
            jointcoords = self.jointcoords_dict[venue_path][session][frames[-1]]
        except:
            # Substitute dummy coords
            logging.error("jointcoords_dict not found!!!")
            logging.error(f"{venue_path=}")
            logging.error(f"{session=}")
            logging.error(f"{frames}")
            jointcoords = np.ones((3, 17)) * np.nan

        direction, _ = hand2marker_direction(
            jointcoords,
            self.markerdict[venue_path],
            action_id,
            self.cameraparam_dict[venue_path][f"{camera_id:02d}"],
            lr,
        )
        answer = {
            "action": self.rev_action_class_dict[action_id],
            "direction": direction,
            "action_id": action_id,
        }

        # If verbose:
        # - load image for visualization
        # - return camera parameters
        image = []
        camera_param = []
        if self.cfg.verbose:
            image_path = (
                Path(venue_path)
                / session
                / f"{camera_id:02d}"
                / f"{frames[-1]+1:010d}.jpg"  # The output of ffmpeg is 1-indexed
            )
            image = torchvision.io.read_image(str(image_path))

            camera_param = self.cameraparam_dict[venue_path][f"{camera_id:02d}"]
            camera_param["filepath"] = str(
                camera_param["filepath"]
            )  # PosixPathはcollateできないので文字列に変える

        return {
            "idx": idx,
            "venue_path": venue_path,
            "session": session,
            "joint_id": joint_id,
            "abs_joint_position": abs_joint_pos,
            "rel_joint_position": rel_joint_pos,
            "images": images,
            "featvec_joint": featvec_joints,
            "featvec_bb": featvec_bbs,
            "featvec_img": featvec_imgs,
            "answer": answer,
            "src_key_padding_mask": src_key_padding_mask,
            "lr": lr,
            "bbox": bbox,
            "has_uncertain_frame": has_uncertain_frame,
            "keypoints": keypoints,
            "jointcoords": jointcoords,
            "camera_id": camera_id,
            "camera_param": camera_param,
            "frames": frames,
        }

    def __len__(self):
        return len(self.datalist)


def hand2marker_direction(
    joints, markers, marker_id, cam_params, lr, number_location=0.6
) -> tuple[np.ndarray, np.ndarray]:
    """
    Given 3D joint coordinates, 3D marker coordinates, and the target marker id,
    calculates the gaze360-like camera coordinate direction from the hand to the marker.
    x=returned[0] is rightwards, y=returned[1] is depthwards, z=returned[2] is upwards

    Params:
        joints: np.ndarray(17,3)
        number_location: Location of the target number. If the target is at the center of the ArUco marker it is zero, and if it's at the top of the marker, it's 0.5. Default value is 0.6, which means the target is a bit above the marker.

    Returns:
        direction_camera: np.ndarray
        direction_world: np.ndarray
    """
    joints = joints.T
    valid_ids = markers["valid_ids"]
    marker_corner = markers["corners"]
    marker_center = marker_corner.mean(axis=1)
    marker_b2t = (
        marker_corner[:, 0]
        + marker_corner[:, 1]
        - marker_corner[:, 2]
        - marker_corner[:, 3]
    ) / 2
    marker_number_coord = marker_center + marker_b2t * number_location
    if lr == "l":
        elbow_id = 7
        hand_id = 9
    else:
        elbow_id = 8
        hand_id = 10

    if marker_id not in valid_ids:
        return np.ones(3) * np.nan, np.ones(3) * np.nan

    # preparation
    hand_coord = joints[hand_id] + 0.2 * (joints[hand_id] - joints[elbow_id])
    cam_coord = cam_params["t"]
    cam_to_hand = hand_coord - cam_coord

    def normalized(x):
        return x / np.linalg.norm(x)

    # Define coordinate systems
    Ez = normalized(cam_to_hand)
    Ey = normalized(np.cross(cam_params["R"][:, 0], Ez))
    if Ey @ cam_params["R"][:, 1] > 0:
        # Image y (=cam_params['R'][:, 1]) is downwards in the image plane and Ey is upwards.
        # They should have a negative dot product.
        Ey = -Ey
    Ex = np.cross(Ey, Ez)
    if Ex @ cam_params["R"][:, 0] < 0:
        # Image x (=cam_params['R'][:,0]) is rightwards in the image plane and Ex is too.
        # They should have a positive dot product.
        Ex = -Ex
    E = np.array((Ex, Ey, Ez))
    #  assert np.isclose(np.linalg.norm(Ex), 1), f"{np.linalg.norm(Ex)=}, {E=}"

    # calculate direction
    direction_world = normalized(marker_number_coord[marker_id] - hand_coord)
    direction_camera = E @ direction_world
    # NOTE: swap y and z so that x is rightward, y=direction[1] is depthward, z=direction[2] is upward
    direction_camera[1], direction_camera[2] = direction_camera[2], direction_camera[1]

    return direction_camera, direction_world


class MovieDataset(Dataset):
    EMPTY_DETECTION = {
        "keypoints": [0 for _ in range(17 * 3)],
        "bbox": [0 for _ in range(4)],
        "score": 0,
        "category_id": 1,
    }
    CATEGORY2ID: dict[str, int] = {
        "pad": 0,
        "cls": 1,
        "nose": 2,
        "left_eye": 3,
        "right_eye": 4,
        "left_ear": 5,
        "right_ear": 6,
        "left_shoulder": 7,
        "right_shoulder": 8,
        "left_elbow": 9,
        "right_elbow": 10,
        "left_wrist": 11,
        "right_wrist": 12,
        "left_hip": 13,
        "right_hip": 14,
        "left_knee": 15,
        "right_knee": 16,
        "left_ankle": 17,
        "right_ankle": 18,
    }
    JOINT_THRESH: float = 0.4
    SCORE_THRESH: float = 0.4
    NUM_JOINTS: int = 17
    SKIPBY = 1

    def __init__(self, movie_path: str, lr: str, token_length: int, device: str):
        self.lr = lr
        self.device = device
        self.cap = cv2.VideoCapture(movie_path)
        self.token_length = token_length
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS), 2)
        self.num_frames = int(round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT), 2))
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
        print(f"{self.fps=}, {self.num_frames=}, {self.width=}, {self.height=}")
        import openpifpaf

        self.predictor = openpifpaf.Predictor(
            visualize_image=True, checkpoint="shufflenetv2k16", json_data=False
        )

        self.predicted_joints = {}
        self.prev_frame = -1

    def __len__(self):
        return (self.num_frames - self.token_length) // self.SKIPBY

    def __getitem__(self, idx: int):
        images = []
        keypoints, bboxes = [], []
        SHRINKBY = 2
        for i in range(idx, idx + self.token_length * self.SKIPBY, self.SKIPBY):
            if i != self.prev_frame:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.prev_frame = i
            ret, frame = self.cap.read()

            # PROCESS BY PIFPAF
            images.append(frame.astype(float) / 255)
            # If not cached, process and cache joints
            if i not in self.predicted_joints:
                # SHRINK IMAGE
                predictions, gt_anns, image_meta = self.predictor.pil_image(
                    Image.fromarray(
                        cv2.resize(
                            frame,
                            (frame.shape[1] // SHRINKBY, frame.shape[0] // SHRINKBY),
                        )
                    )
                )
                try:
                    pred_jsons = predictions[0].json_data()
                except IndexError:
                    pred_jsons = {
                        "keypoints": [0 for _ in range(17 * 3)],
                        "bbox": [0 for _ in range(4)],
                        "score": 0,
                        "category_id": 1,
                    }
                for k in range(17):
                    if pred_jsons["keypoints"][k * 3 + 2] != 0:
                        pred_jsons["keypoints"][k * 3 + 0] *= SHRINKBY
                        pred_jsons["keypoints"][k * 3 + 1] *= SHRINKBY
                self.predicted_joints[i] = pred_jsons
            k = np.array(self.predicted_joints[i]["keypoints"]).reshape((17, 3))
            keypoints.append(k)
            b = np.array(self.predicted_joints[i]["bbox"])
            bboxes.append(b)

        images = rearrange(
            torch.tensor(np.array(images)[:, :, :, ::-1].copy()).float(),
            "n h w c -> n c h w",
        )
        keypoints = torch.tensor(np.array(keypoints)).float()
        bboxes = torch.tensor(np.array(bboxes)).float()
        width, height = bboxes[:, 2], bboxes[:, 3]

        joint_id = [
            torch.where(keypoints[i, :, 2] > self.JOINT_THRESH)[0]
            for i in range(self.token_length)
        ]

        # For simplicity, use the last token as the joint special token
        pad_jointid_tensor = torch.full((self.NUM_JOINTS + 1,), self.CATEGORY2ID["pad"])
        pad_jointid_tensor[-1] = self.CATEGORY2ID["cls"]
        joint_id = pad_sequence(joint_id, pad_tensor=pad_jointid_tensor)

        # joint position tokens
        abs_joint_pos = keypoints[:, :, :2]
        chest_coord = (keypoints[:, 5, :2] + keypoints[:, 6, :2])[:, None, :] / 2
        # Both are supposed to be in range of [-1, 1]
        rel_joint_pos = (abs_joint_pos - chest_coord) / torch.stack(
            [width, height], dim=-1
        )[:, None, :]

        # Add special token to joint position tokens
        rel_joint_pos = torch.concat(
            (rel_joint_pos, torch.tensor([0]).repeat(self.token_length, 1, 2)),
            dim=1,
        )

        # src_key_padding_mask for transformer
        src_key_padding_mask = joint_id == self.CATEGORY2ID["pad"]

        # RETURN
        return {
            "idx": idx,
            "joint_id": joint_id.to(self.device),
            "abs_joint_position": abs_joint_pos.to(self.device),
            "rel_joint_position": rel_joint_pos.to(self.device),
            "images": images.to(self.device),
            "src_key_padding_mask": src_key_padding_mask.to(self.device),
            "orig_image": frame,
            "lr": self.lr,
            "keypoints": keypoints,
            "bbox": bboxes,
        }
