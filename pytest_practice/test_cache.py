import cache

def test_should_refresh_when_expired(mocker):
    # TODO: freeze "now" at 1000.0
    # TODO: call should_refresh(last_fetched_at=980.0, ttl_seconds=20) -> True
    
    # indicate what the mock is going to mock
    mocker.patch('cache.time.time', return_value = 1000.0)

    # call the actual function
    assert cache.should_refresh(980.0, ttl_seconds=20) is True


def test_should_not_refresh_before_ttl(mocker):
    # TODO: freeze "now" at 999.9
    # TODO: call should_refresh(last_fetched_at=980.0, ttl_seconds=20) -> False
    mocker.patch('cache.time.time', return_value = 999.9)

    assert cache.should_refresh(980.0,20) is False

def test_boundary_exactly_on_ttl(mocker):
    # TODO: freeze "now" at 1000.0
    # should_refresh(last_fetched_at=980.0, ttl_seconds=20) -> True
    mocker.patch('cache.time.time', return_value=1000.1)
    assert cache.should_refresh(980.0,20) is True 
