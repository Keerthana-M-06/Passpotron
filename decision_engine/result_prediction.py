
def predict_final_result(minifasnet_result, ear_passed, challenge_status):
    """
    Combines MiniFASNet, EAR and random challenge results.

    Returns:
        LIVE CANDIDATE
        SPOOF SUSPECTED
        INCONCLUSIVE
    """

    # MiniFASNet has detected a possible spoof.
    if minifasnet_result == "SPOOF SUSPECTED":
        return "SPOOF SUSPECTED"

    # Both additional liveness checks must pass.
    if (
        minifasnet_result in ["LIVE CANDIDATE", "INCONCLUSIVE"]
        and ear_passed is True
        and challenge_status == "PASSED"
    ):
        return "LIVE CANDIDATE"

    # Missing, failed or uncertain checks.
    return "INCONCLUSIVE"
