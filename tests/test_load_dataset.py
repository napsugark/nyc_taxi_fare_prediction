import pytest

from src.nyc_tfp.load_data import download_kaggle_competition_data


def test_download_kaggle_competition_data(mocker):
    mocker.patch.dict("os.environ", {"KAGGLE_USERNAME": "test_user", "KAGGLE_KEY": "test_key"})

    mock_api_cls = mocker.patch("src.nyc_tfp.load_data.KaggleApi")
    mock_api = mock_api_cls.return_value
    mock_api.authenticate.return_value = None
    mock_api.competition_download_files.return_value = None

    mock_exists = mocker.patch("src.nyc_tfp.load_data.Path.exists", side_effect=[False, True])

    mock_makedirs = mocker.patch("src.nyc_tfp.load_data.os.makedirs")
    mock_zipfile = mocker.patch("src.nyc_tfp.load_data.zipfile.ZipFile")
    mock_unlink = mocker.patch("src.nyc_tfp.load_data.Path.unlink")

    
    download_kaggle_competition_data("test-competition", "test-path", expected_file="train.csv")

   
    mock_api.authenticate.assert_called_once()
    mock_api.competition_download_files.assert_called_once_with("test-competition", path="test-path")
    mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with("test-path")
    mock_makedirs.assert_called_once_with("test-path", exist_ok=True)
    mock_unlink.assert_called_once()


