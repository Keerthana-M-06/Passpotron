
class RetryManager:

    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.attempts_used = 0
        self.attempt_active = False

    def start_attempt(self):
        """
        Start one liveness verification attempt.
        Called only when a valid face is detected
        and a random challenge is started.
        """

        if self.attempt_active:
            return self.attempts_used

        if self.attempts_used >= self.max_attempts:
            return None

        self.attempts_used += 1
        self.attempt_active = True

        print(
            f"Starting Trial {self.attempts_used}"
            f" of {self.max_attempts}"
        )

        return self.attempts_used

    def finish_attempt(self, result):
        """
        Process the final liveness result.
        """

        self.attempt_active = False

        if result == "LIVE CANDIDATE":
            return "PROCEED"

        if result == "SPOOF SUSPECTED":
            return "FLAG FOR REVIEW"

        if result == "INCONCLUSIVE":

            if self.attempts_used < self.max_attempts:
                return "RETRY"

            return "MANUAL REVIEW"

        return "MANUAL REVIEW"

    def interrupt_attempt(self):
        """
        A face disappears after an attempt has started.
        Treat the interrupted attempt as inconclusive.
        """

        if not self.attempt_active:
            return None

        return self.finish_attempt("INCONCLUSIVE")

    def reset(self):
        """
        Reset only when starting a completely new
        verification session.
        """

        self.attempts_used = 0
        self.attempt_active = False
