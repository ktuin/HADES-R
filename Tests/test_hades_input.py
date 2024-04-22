import unittest
from HadesR import hades_input
from HadesR.hades_utils import distance_calculation


class TestHadesInput(unittest.TestCase):
    def setUp(self):
        self.data_path = "/path/to/data"
        self.event_file = "event.dat"
        self.station_file = "stations.txt"
        self.sta_select = ["station1", "station2"]
        self.hades_input = hades_input(
            self.data_path, self.event_file, self.station_file, self.sta_select
        )

    # TODO Add more


if __name__ == "__main__":
    unittest.main()
