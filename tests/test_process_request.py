from seer.db import ProcessRequest, Session


def test_create_process_request():

def test_process_request_rapid_cud_operations():
    try:
        for _ in range(100):  # Simulating rapid CUD operations
            with Session() as session:
                process_request = ProcessRequest(name=\"rapid-cud-test\", payload={\"key\": \"value\"})
                session.add(process_request)
                session.commit()  # Committing the create operation
                
                process_request.payload = {\"key\": \"updated value\"}
                session.commit()  # Committing the update
                
                session.delete(process_request)
                session.commit()  # Committing the delete
                
    except Exception as e:
        assert False, f\"Unexpected exception: {e}\"
    else:
        assert True, \"Rapid CUD operations executed successfully\"
    with Session() as session:
        session.add(ProcessRequest(name="test-name", payload={}))
        session.commit()
