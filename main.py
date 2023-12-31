#!/usr/bin/env python3

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import cv2
import pygame
import sys
import numpy as np
from time import time
import threading
import mediapipe as mp
from glob import glob
import csv

pygame.init()
pygame.font.init()

game_path = os.path.dirname(os.path.realpath(__file__))

# Some important relative paths
# TODO : Get Pokemon like mask
BATTLE_PATH = os.path.join(game_path, "assets/battle/battle.mp3")
CRY_PATH = os.path.join(game_path, "assets/cries/")
MASK_PATH = os.path.join(game_path, "assets/masks/coords.png")
OPTIONS_PIC_PATH = os.path.join(game_path, "assets/battle/battle.png") 
HAT_ANNOT_PATH = os.path.join(game_path, "assets/masks/hat_labels.csv") 

# Settings for test 
DEFAULT_RES = [1920,1080]
DEFAULT_FPS = 30
DELAY_START_TIME = 10
STAT_FREQUENCY = 3
FLASH_FREQUENCY = 15
FLASH_DURATION = 3
BLACK_DURATION = 3
RUN_CLASSIFIER = True
RUN_OPTICAL_FLOW = True
CRY_DELAY = 7.0
CLASSIFIER_DELAY = 0.1
DISPLAY_FPS = False
RUN_BATTLE = True
VISUALIZE_POINTS = False
HAT_KEYPOINTS = [54, 67, 10, 297, 284]
ADD_MASK = True

class Camera():
    """ The dumb game camera / classifier class. Sets masks, gets frames, looks at your face. """
    def __init__(self):
        """ Init and load parameters """
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_RES[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_RES[1])
        self.cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS) 
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.init_battle = True
        self.eye_cache = None

        # Classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.relevant_keypoints = []

        # Optical Flow
        self.first_frame = True
        self.lk_params = { 
            "winSize" : (101, 101),
            "maxLevel" : 15,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001)
        }
        self.points_to_prev = None
        self.sigma_dist = 50

        # FPS
        self.count = 0
        self.t_start = time()
        self.t_frm_start = 0
        self.t_elapsed = 0
        self.stat_freq = STAT_FREQUENCY
        self.avg_fps = 0

        # Set buffers
        self.load_mask()
        self.load_frame()
        self.frame_h, self.frame_w, _ = self.img.shape
        self.pyg_buffer = np.zeros([self.frame_w, self.frame_h, 3], dtype=np.uint8)

    def load_frame(self):
        """ Load camera frame """
        if(self.cap.isOpened()):
            self.rv, self.img = self.cap.read()
            if not self.rv:
                return None

            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

            # TODO: Set all of the things in the frame...
            self.frame = self.img

    def load_mask(self, mask_path=MASK_PATH, hat_annot_path=HAT_ANNOT_PATH):
        """ Switching to mediapipe because fast """
        # Read the image
        img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        b, g, r, self.filter_alpha = cv2.split(img)
        self.filter_mask = cv2.merge((b, g, r))

        self.mask_keypoints = {}
        with open(hat_annot_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for _, row in enumerate(csv_reader):
                # skip head or empty line if it's there
                try:
                    x, y = int(row[1]), int(row[2])
                    self.mask_keypoints[row[0]] = (x, y)
                except ValueError:
                    continue


    def set_mask(self, mask_path):
        """ Make some transparent buffers to add mask every frame"""
        face_mask = cv2.imread(mask_path, -1)
        _, _, channels = face_mask.shape
        if channels < 4:
            face_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2BGRA)

        gray_mask = cv2.cvtColor(face_mask, cv2.COLOR_BGR2GRAY)
        _ , thresh = cv2.threshold(gray_mask, 240, 255, cv2.THRESH_BINARY)
        face_mask[thresh == 255] = 0

        # make alpha, bound mask, and make smallest image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(cnts, key=cv2.contourArea)[-1]
        x,y,w,h = cv2.boundingRect(cnt)
        face_mask = face_mask[y:y+h, x:x+w]
        gray_mask = gray_mask[y:y+h, x:x+w]
        self.thresh_mask = thresh[y:y+h, x:x+w]

        # Remove background
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(self.thresh_mask, cv2.MORPH_CLOSE, kernel)
        roi, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(face_mask.shape, face_mask.dtype)
        cv2.fillPoly(mask, roi, (255,) * face_mask.shape[2])
        self.mask = cv2.bitwise_and(face_mask, mask)
        frame_w, frame_h, _ = self.mask.shape
        self.mask_src_mat = np.array([
            [0,0], 
            [frame_w, 0],  
            [frame_w, frame_w], 
            [0, frame_h]
        ])


    def people_finder(self):
        """ Face / Eyes classifier and add mask """
        faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # roi = self.gray[y : y + h, x : x + w]
            # eyes = self.eye_cascade.detectMultiScale(roi)

            dst_mat = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ])

            hom = cv2.findHomography(self.mask_src_mat, dst_mat)[0]
            warped = cv2.warpPerspective(self.mask, hom, (self.frame_w, self.frame_h))
            w_mask = warped[:,:,3]

            # Copy and convert the mask to a float and give it 3 channels
            mask_scale = w_mask.copy() / 255.0
            mask_scale = np.dstack([mask_scale] * 3)
            warped = cv2.cvtColor(warped, cv2.COLOR_BGRA2BGR)
            warped_multiplied = cv2.multiply(mask_scale, warped.astype("float"))
            # TODO: Something is wrong here with mask_scale. need to figure it out.
            image_multiplied = cv2.multiply(self.frame.astype(float), 1.0 - mask_scale)
            self.frame = cv2.add(warped_multiplied, image_multiplied)
            self.frame = self.frame.astype("uint8")

    def similarityTransform(self, inPoints, outPoints):
        s60 = np.sin(60*np.pi/180)
        c60 = np.cos(60*np.pi/180)

        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()

        # The third point is calculated so that the three points make an equilateral triangle
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

        inPts.append([int(xin), int(yin)])

        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

        outPts.append([int(xout), int(yout)])


        # Now we can use estimateRigidTransform for calculating the similarity transform.
        tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
        return tform[0]

    def mp_face_finder(self):
        selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                     285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                     387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                     178, 162, 54, 67, 10, 297, 284, 389]

        results = self.face_mesh.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            return

        for landmarks in results.multi_face_landmarks:
            values = np.array(landmarks.landmark)
            keypoints = np.zeros((len(values), 2))

            for idx ,value in enumerate(values):
                keypoints[idx][0] = value.x
                keypoints[idx][1] = value.y

            keypoints = keypoints * (self.frame_w, self.frame_h)
            keypoints = keypoints.astype('int')

            self.relevant_keypoints = []

            for ii in HAT_KEYPOINTS:
                self.relevant_keypoints.append(keypoints[ii])

        if RUN_OPTICAL_FLOW:
            if self.first_frame:
                self.first_frame = False
                self.points_to_prev = np.array(self.relevant_keypoints, np.float32)
                self.gray_prev = np.copy(self.gray)

            self.points_to_next, st, err = cv2.calcOpticalFlowPyrLK(self.gray_prev, self.gray, self.points_to_prev, np.array(self.relevant_keypoints, np.float32), **self.lk_params)

            # Average landmark points with detections and tracked points
            for ii in range(0, len(self.relevant_keypoints)):
                point = self.relevant_keypoints[ii]
                next_point = self.points_to_next[ii]
                dist = cv2.norm(np.array(point - next_point))
                alpha = np.exp(-dist * dist / self.sigma_dist)
                point = (1 - alpha) * np.array(point) + alpha * next_point
                point = (min(max(point[0], 0), self.frame_w - 1), min(max(point[1], 0), self.frame_h - 1))
                self.relevant_keypoints[ii] = (int(point[0]), int(point[1]))

            self.points_to_prev = np.array(self.relevant_keypoints, np.float32)
            self.gray_prev = self.gray

        if VISUALIZE_POINTS:
            for idx, point in enumerate(self.relevant_keypoints):
                cv2.circle(self.frame, point, 2, (255, 0, 0), -1)
                cv2.putText(self.frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)

        # Apply similarity transform to input image
        tform = self.similarityTransform(list(self.mask_keypoints.values()), self.relevant_keypoints)
        trans_img = cv2.warpAffine(self.filter_mask, tform, (self.frame_w, self.frame_h))
        trans_alpha = cv2.warpAffine(self.filter_alpha, tform, (self.frame_w, self.frame_h))
        mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))
 
        # Blur the mask before blending
        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)
        mask2 = (255.0, 255.0, 255.0) - mask1
 
        # Perform alpha blending of the two images
        temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
        temp2 = np.multiply(self.frame, (mask2 * (1.0 / 255)))
        if ADD_MASK:
            self.frame = np.uint8(temp1 + temp2)



    def update_stats(self):
        """ Update fps stats """
        self.count += 1

        if (self.count % self.stat_freq == 0):
            self.t_elapsed = time() - self.t_start
            t2 = time()
            self.avg_fps = self.stat_freq / (t2 - self.t_frm_start)
            self.t_frm_start = t2


    def to_pygame_frame(self):
        """ convert CV Mat to pygame frame """
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # TODO: make buffer to avoid overhead
        self.frame = cv2.transpose(self.frame)
        self.frame = cv2.flip(self.frame, 0)


class Window():
    """ The dumb game pygame window. Load images and sounds and trigger some events... """
    def __init__(self, cam:Camera, res:list[int] = DEFAULT_RES):
        print("Gotta catch em all..")
        self.cam = cam
        self.res = res
        self.mixer = pygame.mixer
        self.mixer.init(44100, -16, 1, 1024)
        self.load_assets()
        self.fps = cam.fps
        self.clock = pygame.time.Clock()
        self.clock.tick(cam.fps)

        pygame.display.set_caption("Loading Cinnabar Island")
        pygame.display.set_mode(vsync=1)
        self.fps_disp = pygame.font.SysFont('Comic Sans MS', 30)
        self.screen = pygame.display.set_mode(res)
        self.screen.fill([0,0,0])
        self.init_battle = True
        self.init_cry = False
        self.fading = False
        self.alpha = 0
        self.alpha_delta = int(255 / (self.fps / FLASH_FREQUENCY))


    def load_assets(self, sound_path=BATTLE_PATH, mask_path=MASK_PATH, cry_path=CRY_PATH, fight_template=OPTIONS_PIC_PATH):
        """ Load pokemon crying and other things """
        cry_files = glob(pathname= os.path.join(cry_path, "*.mp3"))
        pid = np.random.randint(low=0, high=len(cry_files))
        cry_file = os.path.join(cry_path, cry_files[pid])
        assert(os.path.exists(cry_file))
        self.cry_sound = self.mixer.Sound(cry_file)

        self.battle_sound = self.mixer.Sound(sound_path)
        self.cam.set_mask(mask_path)
        self.fight_image = pygame.image.load(fight_template)
        self.fight_image = pygame.transform.scale(self.fight_image, (self.res[1], 400))
        self.fight_rect = self.fight_image.get_rect()


    def draw_fade(self, frame):
        """ Alpha flashing """
        if self.fading:
            self.alpha += self.alpha_delta
            if self.alpha <= 0 or self.alpha >= 255:
                self.alpha_delta *= -1
            frame.set_alpha(self.alpha)
        return frame


    def draw_black(self, frame):
        """ Paint it black """
        self.alpha = 0
        frame.set_alpha(self.alpha)
        return frame


    def start_cry(self): 
        """ Use of POSIX threads to play background and crying sounds """
        t = threading.Thread(name = 'cry_sound', target = self.cry_sound.play)
        t.daemon = True
        t.start()
        t.join()


    def start_battle(self): 
        """ Start music, animations, and menu """
        t = threading.Thread(name = 'battle_sound', target = self.battle_sound.play, args=(-1,))
        t.daemon = True
        t.start()
        t.join()


    def run_game(self):
        """ Become a pokemon master """
        while True:
            ftime = time()
            self.screen.fill(pygame.Color('black'))
            self.cam.load_frame()

            if RUN_CLASSIFIER and self.cam.t_elapsed > DELAY_START_TIME + FLASH_DURATION + BLACK_DURATION + CLASSIFIER_DELAY:
                #self.cam.people_finder()
                self.cam.mp_face_finder()

            self.cam.update_stats()

            self.cam.to_pygame_frame()
            frame = pygame.surfarray.make_surface(self.cam.frame)

            if RUN_BATTLE:
                # Start Battle
                if self.cam.t_elapsed > DELAY_START_TIME and self.init_battle:
                    self.t_battle_start = time()
                    self.init_battle = False
                    self.init_cry = True
                    self.fading = True
                    self.start_battle()

                if self.fading == True:
                    if (ftime - self.t_battle_start) <= FLASH_DURATION:
                        frame = self.draw_fade(frame)
                    else:
                        battle_dur = ftime - self.t_battle_start
                        if (battle_dur > FLASH_DURATION) and (battle_dur < FLASH_DURATION + BLACK_DURATION):
                            self.draw_black(frame)
                        else:
                            self.fading = False

                if self.init_cry:
                    if (ftime - self.t_battle_start > CRY_DELAY):
                        self.init_cry = False
                        self.start_cry()

                self.screen.blit(frame, (0,0))

                if self.cam.t_elapsed > DELAY_START_TIME + FLASH_DURATION + BLACK_DURATION:
                    scene_y = frame.get_height()
                    self.screen.blit(self.fight_image, (0, scene_y - self.fight_rect[3]))


                if DISPLAY_FPS:
                    text_surface = self.fps_disp.render(f"FPS: {self.cam.avg_fps:.1f}. Play time: {self.cam.t_elapsed:.1f}", False, (0, 0, 0))
                    self.screen.blit(text_surface, (0,0))
            else:
                self.screen.blit(frame, (0,0))

            pygame.display.update()


            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    sys.exit(0)


def game_pipeline():
    r""" ¯\_(ツ)_/¯ """
    cam = Camera()
    win = Window(cam=cam)
    win.run_game()
    pygame.quit()


if __name__ == "__main__":
    game_pipeline()
