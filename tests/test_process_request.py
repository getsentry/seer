from seer.db import ProcessRequest, Session


def test_create_process_request():
    with Session() as session:
        session.add(ProcessRequest(name="test-name", payload={}))
        session.commit()
