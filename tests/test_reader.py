import os
import pytest
import tempfile
import zipfile
from pathlib import Path
from campaign_reader import CampaignReader, CampaignZipError

@pytest.fixture
def sample_zip():
    """Create a temporary zip file with test content."""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zf:
            # Add some test files
            zf.writestr('test1.txt', 'Test content 1')
            zf.writestr('folder/test2.txt', 'Test content 2')
            zf.writestr('config.json', '{"test": "data"}')
        
        yield temp_zip.name
        # Cleanup after tests
        if os.path.exists(temp_zip.name):
            os.unlink(temp_zip.name)

@pytest.fixture
def bad_zip():
    """Create an invalid zip file."""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        temp_zip.write(b'Not a zip file')
        temp_zip.flush()
        yield temp_zip.name
        if os.path.exists(temp_zip.name):
            os.unlink(temp_zip.name)

def test_zip_file_not_found():
    """Test handling of non-existent zip file."""
    with pytest.raises(FileNotFoundError):
        CampaignReader('nonexistent.zip')

def test_invalid_zip_file(bad_zip):
    """Test handling of invalid zip file."""
    with pytest.raises(CampaignZipError):
        CampaignReader(bad_zip)

def test_successful_zip_reading(sample_zip):
    """Test successful reading of a valid zip file."""
    reader = CampaignReader(sample_zip)
    try:
        # Check if files were extracted
        assert len(reader.list_files()) == 3
        assert 'test1.txt' in reader.list_files()
        assert 'folder/test2.txt' in reader.list_files()
        assert 'config.json' in reader.list_files()
    finally:
        reader.cleanup()

def test_extracted_file_access(sample_zip):
    """Test accessing extracted files."""
    with CampaignReader(sample_zip) as reader:
        # Get path to extracted file
        test1_path = reader.get_extracted_path('test1.txt')
        assert test1_path is not None
        assert test1_path.exists()
        
        # Read content
        with open(test1_path) as f:
            content = f.read()
        assert content == 'Test content 1'

def test_context_manager_cleanup(sample_zip):
    """Test that files are cleaned up when using context manager."""
    extracted_paths = []
    with CampaignReader(sample_zip) as reader:
        # Store paths for later checking
        for fname in reader.list_files():
            path = reader.get_extracted_path(fname)
            assert path.exists()
            extracted_paths.append(path)
    
    # Verify cleanup
    for path in extracted_paths:
        assert not path.exists()

def test_custom_extract_dir(sample_zip):
    """Test extraction to custom directory."""
    temp_dir = tempfile.mkdtemp()
    try:
        with CampaignReader(sample_zip, extract_dir=temp_dir) as reader:
            # Check if files are in custom directory
            test1_path = reader.get_extracted_path('test1.txt')
            assert str(test1_path).startswith(temp_dir)
            assert test1_path.exists()
    finally:
        # Clean up the temp directory manually
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.unlink(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)

def test_cleanup_after_extraction_error(sample_zip):
    """Test cleanup after failed extraction."""
    # Create a temporary directory that we'll manage manually
    temp_dir = tempfile.mkdtemp()
    try:
        # Make the directory readonly
        os.chmod(temp_dir, 0o444)
        
        with pytest.raises(CampaignZipError):
            CampaignReader(sample_zip, extract_dir=temp_dir)
        
        # Make the directory writable again so we can check and clean it
        os.chmod(temp_dir, 0o755)
        
        # Verify no files were left behind
        assert len(os.listdir(temp_dir)) == 0
    finally:
        # Ensure cleanup
        if os.path.exists(temp_dir):
            os.chmod(temp_dir, 0o755)  # Ensure we can delete it
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.unlink(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)

def test_malicious_zip_paths():
    """Test handling of potentially malicious zip paths."""
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip.name, 'w') as zf:
            # Try to write file outside extraction directory
            zf.writestr('../outside.txt', 'Bad content')
        
        with pytest.raises(CampaignZipError):
            CampaignReader(temp_zip.name)
        
        os.unlink(temp_zip.name)