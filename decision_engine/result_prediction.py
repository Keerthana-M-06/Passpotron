
def predict_final_result(
    minifasnet_result,
    ear_passed,
    challenge_status
):

    # A confirmed MiniFASNet spoof suspicion takes priority.
    if minifasnet_result == "SPOOF SUSPECTED":
        return "SPOOF SUSPECTED"

    # All three checks must pass.
    if (
        minifasnet_result == "LIVE CANDIDATE"
        and ear_passed is True
        and challenge_status == "PASSED"
    ):
        return "LIVE CANDIDATE"

    # Any missing, failed or inconclusive check.
    return "INCONCLUSIVE"
