
import random
import time


class RandomChallenge:

    def __init__(self):

        self.challenges = [
            "BLINK_TWICE",
            "TURN_LEFT",
            "TURN_RIGHT",
            "CLOSE_EYES"
        ]

        self.current_challenge = None
        self.start_time = None

        self.time_limit = 8
        self.close_eyes_duration = 0.7

        self.completed = False
        self.failed = False

        self.initial_blink_count = 0
        self.eyes_closed_since = None

    def start(self, blink_count):

        self.current_challenge = random.choice(
            self.challenges
        )

        self.start_time = time.monotonic()

        self.initial_blink_count = blink_count

        self.completed = False
        self.failed = False
        self.eyes_closed_since = None

        print(
            "New challenge:",
            self.current_challenge,
            flush=True
        )

        return self.current_challenge

    def get_instruction(self):

        instructions = {
            "BLINK_TWICE": "Please blink twice",
            "TURN_LEFT": "Please turn your head LEFT",
            "TURN_RIGHT": "Please turn your head RIGHT",
            "CLOSE_EYES": "Close your eyes briefly"
        }

        return instructions.get(
            self.current_challenge,
            "Waiting for challenge"
        )

    def update(
        self,
        blink_count,
        head_direction,
        eyes_closed
    ):

        if self.current_challenge is None:
            return "NOT_STARTED"

        if self.completed:
            return "PASSED"

        if self.failed:
            return "FAILED"

        # Check the 8-second timeout.
        if time.monotonic() - self.start_time > self.time_limit:

            self.failed = True
            return "FAILED"

        # Challenge 1: Blink twice.
        if self.current_challenge == "BLINK_TWICE":

            if blink_count - self.initial_blink_count >= 2:
                self.completed = True

        # Challenge 2: Turn head left.
        elif self.current_challenge == "TURN_LEFT":

            if head_direction == "LEFT":
                self.completed = True

        # Challenge 3: Turn head right.
        elif self.current_challenge == "TURN_RIGHT":

            if head_direction == "RIGHT":
                self.completed = True

        # Challenge 4: Close eyes for at least 0.7 seconds.
        elif self.current_challenge == "CLOSE_EYES":

            if eyes_closed:

                if self.eyes_closed_since is None:
                    self.eyes_closed_since = time.monotonic()

                elif (
                    time.monotonic() - self.eyes_closed_since
                    >= self.close_eyes_duration
                ):
                    self.completed = True

            else:
                # Reset if the user opens their eyes too early.
                self.eyes_closed_since = None

        if self.completed:
            return "PASSED"

        return "IN_PROGRESS"
