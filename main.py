from libchocolatine.libchocolatine import ChocolatineDetector
from libchocolatine.asyncfetcher import AsyncHistoryFetcher
from constants import name, apiurl, kafkaconf, dbconf


def start_detector_and_fetcher():
    """
    Starts the detector and fetcher based on the provided API URL, Kafka configuration and
    PostgreSQL DB configuration.

    :return: The detector and fetcher objects for use in other functions to obtain results.
    """
    det = ChocolatineDetector(name, apiurl, kafkaconf, dbconf)
    fetcher = AsyncHistoryFetcher(apiurl, det.histRequest, det.histReply)

    det.start()
    fetcher.start()

    return det, fetcher


def fetch_result(det, key, timestamp, value):
    """
    Adds live data to the queue and obtain live data results.

    :param det: The detector object.
    :param key: The key for the live data.
                It consists of two parts: the first is either of "bgp", "google_tr"/"gtr", "darknet" or "active".
                The second is ... #TODO
    :param timestamp: The timestamp for the live data in the format: #TODO
    :param value: The value of the live data. (?) #TODO
    :param blocking: Whether to block and wait for the result (default is False).
    :return: The result of the live data fetch.
    """
    det.queueLiveData(key, timestamp, value)
    res = det.getLiveDataResult(block=False)

    print(res)


def main():
    print("Starting app...")
    det, fetcher = None, None
    try:
        det, fetcher = start_detector_and_fetcher()
    except Exception as e:
        print(f"An error occurred: {e}")

    key = "bgp.as1234"
    timestamp = "2025-02-10T12:00:00Z"
    value = 42

    fetch_result(det, key, timestamp, value)

    stop_detector(det, fetcher)


def stop_detector(det, fetcher):
    det.halt()
    fetcher.halt()


if __name__ == "__main__":
    main()
