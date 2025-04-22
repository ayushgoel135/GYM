import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pandas as pd
import time
import os
import pygame
from enum import Enum, auto
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# Constants
DEFAULT_FRAME_SIZE = (640, 480)
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
FPS = 30

# Initialize pygame for audio
pygame.mixer.init()

# Initialize MediaPipe Pose once
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()


class ExerciseStage(Enum):
    UP = auto()
    DOWN = auto()
    REST = auto()


@dataclass
class ExerciseConfig:
    joints: Tuple[str, str, str]
    angle_range: Tuple[float, float]
    instructions: str
    completion_threshold: float = 0.9
    rest_threshold: float = 1.1


class AudioManager:
    def __init__(self):
        self.volume = 0.7
        self.enabled = True
        # Default silent sounds if files not found
        self.sounds = {
            "correct": self._create_silent_sound(),
            "incorrect": self._create_silent_sound(),
            "complete": self._create_silent_sound()
        }
        # Try loading actual sound files
        self._load_sounds()

    def _create_silent_sound(self):
        return pygame.mixer.Sound(buffer=bytearray([0] * 1000))

    def _load_sounds(self):
        sound_files = {
            "correct": "good.mp3",
            "incorrect": "beep.mp3",
            "complete": "level_up.mp3"
        }
        for name, path in sound_files.items():
            if os.path.exists(path):
                try:
                    self.sounds[name] = pygame.mixer.Sound(path)
                except:
                    pass

    def play(self, sound_name: str):
        if not self.enabled or sound_name not in self.sounds:
            return
        sound = self.sounds[sound_name]
        sound.set_volume(self.volume)
        sound.play()


audio_manager = AudioManager()


class PoseAnalyzer:
    @staticmethod
    def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    @staticmethod
    def get_landmark_points(landmarks, side: Side, joints: Tuple[str, str, str]) -> Tuple[float, List[float]]:
        prefix = "LEFT_" if side == Side.LEFT else "RIGHT_"
        points = [
            [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[0]}").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[0]}").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[1]}").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[1]}").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[2]}").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{prefix}{joints[2]}").value].y]
        ]
        angle = PoseAnalyzer.calculate_angle(*points)
        return angle, points[1]


class FeedbackSystem:
    def __init__(self):
        self.consecutive_errors = 0

    def analyze_form(self, exercise: str, left_angle: float, right_angle: float,
                     config: ExerciseConfig) -> Tuple[List[str], bool]:
        feedback = []
        critical_error = False

        # Check balance between sides
        if abs(left_angle - right_angle) > 20:
            feedback.append("⚠️ Keep both sides balanced!")
            critical_error = True

        # Check range of motion
        if left_angle < config.angle_range[0] or right_angle < config.angle_range[0]:
            feedback.append(f"⬇️ Increase range! Should be {config.angle_range[0]}-{config.angle_range[1]}°")
            critical_error = True
        elif left_angle > config.angle_range[1] or right_angle > config.angle_range[1]:
            feedback.append(f"⬆️ Reduce range! Should be {config.angle_range[0]}-{config.angle_range[1]}°")
            critical_error = True

        # Update error tracking
        if critical_error:
            self.consecutive_errors += 1
            if self.consecutive_errors >= 2:
                feedback.append("🔴 Multiple form issues detected!")
        else:
            self.consecutive_errors = 0

        return feedback if feedback else ["👍 Perfect form! Keep it up!"], critical_error


EXERCISE_CONFIGS = {
    "Bicep Curl": ExerciseConfig(
        joints=("SHOULDER", "ELBOW", "WRIST"),
        angle_range=(30, 160),
        instructions="Keep elbows close to your body and control the movement.",
        completion_threshold=0.85
    ),
    "Squat": ExerciseConfig(
        joints=("HIP", "KNEE", "ANKLE"),
        angle_range=(70, 160),
        instructions="Keep knees aligned with toes and back straight.",
        completion_threshold=0.8
    )
}


class AIGymTrainer:
    def __init__(self):
        self.cap = None
        self.pose = None
        self.counter = 0
        self.incorrect_counter = 0
        self.stage = ExerciseStage.REST
        self.start_time = None
        self.exercise_data = {
            "Reps": [], "Left Angle": [], "Right Angle": [], "Time": []
        }

    def start_session(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                st.error("Could not open webcam")
                return False

            self.pose = mp_pose.Pose(
                min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            self.start_time = time.time()
            return True

        except Exception as e:
            st.error(f"Error starting session: {str(e)}")
            return False

    def process_frame(self, frame, exercise: str, show_landmarks: bool):
        config = EXERCISE_CONFIGS.get(exercise)
        if not config:
            return frame, ["Invalid exercise selected"]

        try:
            # Process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            feedback = []

            if results.pose_landmarks:
                # Get angles
                left_angle, left_center = PoseAnalyzer.get_landmark_points(
                    results.pose_landmarks.landmark, Side.LEFT, config.joints
                )
                right_angle, right_center = PoseAnalyzer.get_landmark_points(
                    results.pose_landmarks.landmark, Side.RIGHT, config.joints
                )

                # Get feedback
                feedback, critical = FeedbackSystem().analyze_form(
                    exercise, left_angle, right_angle, config
                )

                # Check for rep completion
                avg_angle = (left_angle + right_angle) / 2
                if self._check_rep_completion(avg_angle, config):
                    self._handle_rep_completion(critical)
                    self.exercise_data["Reps"].append(self.counter)
                    self.exercise_data["Left Angle"].append(left_angle)
                    self.exercise_data["Right Angle"].append(right_angle)
                    self.exercise_data["Time"].append(time.time() - self.start_time)

                # Draw landmarks if enabled
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                # Draw angles
                cv2.putText(image, f'L: {int(left_angle)}°',
                            tuple(np.multiply(left_center, DEFAULT_FRAME_SIZE).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'R: {int(right_angle)}°',
                            tuple(np.multiply(right_center, DEFAULT_FRAME_SIZE).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), feedback

        except Exception as e:
            return frame, [f"Processing error: {str(e)}"]

    def _check_rep_completion(self, avg_angle: float, config: ExerciseConfig) -> bool:
        if avg_angle > config.angle_range[1] * config.completion_threshold:
            self.stage = ExerciseStage.DOWN
        elif (avg_angle < config.angle_range[0] * config.rest_threshold and
              self.stage == ExerciseStage.DOWN):
            return True
        return False

    def _handle_rep_completion(self, critical: bool):
        self.stage = ExerciseStage.UP
        #self.counter += 1
        if not critical:
            self.counter += 1
            audio_manager.play("correct")
        else:
            self.incorrect_counter += 1
            audio_manager.play("incorrect")

    def end_session(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.pose:
            self.pose.close()


def main():
    st.set_page_config(page_title="AI Gym Trainer", layout="wide")
    st.title("AI Gym Trainer")

    # Initialize session state
    if 'session_active' not in st.session_state:
        st.session_state.session_active = False
    if 'trainer' not in st.session_state:
        st.session_state.trainer = AIGymTrainer()

    # Sidebar Configuration
    with st.sidebar:
        st.header("Exercise Settings")
        selected_exercise = st.selectbox(
            "Choose Exercise",
            list(EXERCISE_CONFIGS.keys()),
            index=0
        )

        st.header("Workout Parameters")
        target_reps = st.number_input("Target Repetitions", 1, 100, 10)

        st.header("Feedback Settings")
        show_landmarks = st.checkbox("Show Pose Landmarks", True)
        audio_manager.enabled = st.checkbox("Enable Audio Feedback", True)
        audio_manager.volume = st.slider("Volume", 0.0, 1.0, 0.7)

    # Main UI
    video_placeholder = st.empty()
    feedback_placeholder = st.empty()
    stats_placeholder = st.empty()

    # Control buttons
    col1, col2 = st.columns(2)
    if col1.button("Start Session") and not st.session_state.session_active:
        if st.session_state.trainer.start_session():
            st.session_state.session_active = True
            st.rerun()

    if col2.button("Stop Session") and st.session_state.session_active:
        st.session_state.trainer.end_session()
        st.session_state.session_active = False
        st.rerun()

    # Session loop
    if st.session_state.session_active:
        trainer = st.session_state.trainer
        feedback_system = FeedbackSystem()

        while st.session_state.session_active and trainer.counter < target_reps:
            start_time = time.time()

            # Capture frame
            ret, frame = trainer.cap.read()
            if not ret:
                st.warning("Failed to capture frame")
                break

            # Process frame
            frame = cv2.resize(frame, DEFAULT_FRAME_SIZE)
            processed_frame, feedback = trainer.process_frame(
                frame, selected_exercise, show_landmarks
            )

            # Update UI
            video_placeholder.image(processed_frame, channels="BGR")
            feedback_placeholder.markdown("### Feedback\n" + "\n".join(f"• {msg}" for msg in feedback))

            # Update stats
            stats_placeholder.markdown(f"""
                ### Session Stats
                - **Exercise**: {selected_exercise}
                - **Reps**: {trainer.counter}/{target_reps}
                - **Incorrect**: {trainer.incorrect_counter}
                - **Accuracy**: {(trainer.counter - trainer.incorrect_counter) / trainer.counter * 100 if trainer.counter > 0 else 0:.1f}%
            """)

            # Control frame rate
            time.sleep(max(0, 1 / FPS - (time.time() - start_time)))

        # Session complete
        if trainer.counter >= target_reps:
            st.balloons()
            st.success(f"🎉 Completed {target_reps} reps of {selected_exercise}!")
            audio_manager.play("complete")

            # Show analytics
            st.subheader("Session Analytics")
            df = pd.DataFrame(trainer.exercise_data)
            st.line_chart(df[["Left Angle", "Right Angle"]])

            st.session_state.trainer.end_session()
            st.session_state.session_active = False


if __name__ == "__main__":
    main()