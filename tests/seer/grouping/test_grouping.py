import unittest

from seer.db import DbGroupingRecord, Session
from seer.grouping.grouping import (
    BulkCreateGroupingRecordsResponse,
    CreateGroupingRecordData,
    CreateGroupingRecordsRequest,
    GroupingRecord,
    GroupingRequest,
    GroupingResponse,
    SimilarityResponse,
)
from seer.inference_models import grouping_lookup


class TestGrouping(unittest.TestCase):
    def test_get_nearest_neighbors_has_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has a hash and and the request's data matches
        an existing record (ie. the neighbor exists within the threshold)
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace="stacktrace",
            message="message",
            hash="13501807435378261861369456856144",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=1,
                    parent_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
        )

    def test_get_nearest_neighbors_has_neighbor_no_group_id(self):
        """
        Tests get_nearest_neighbors when the request has only a hash and the request's data matches
        an existing record (ie. the neighbor exists within the threshold)
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            project_id=1,
            stacktrace="stacktrace",
            message="message",
            hash="13501807435378261861369456856144",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=None,
                    parent_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
        )

    def test_get_nearest_neighbors_no_neighbor(self):  # TODO jodi: need to test all no group ids
        """
        Test get_nearest_neighbors when no matching record exists. Assert that the record for
        the group hash was added.
        """
        grouping_request = GroupingRequest(
            hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            project_id=1,
            stacktrace="stacktrace",
            message="message",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = (
                session.query(DbGroupingRecord)
                .filter_by(hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
                .first()
            )

        assert new_record
        assert response == SimilarityResponse(responses=[])

    def test_insert_new_grouping_record_group_record_exists(self):
        """
        Tests that insert_new_grouping_record only creates one record per group hash.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()
            # Re-insert the grouping record
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()
            matching_record = (
                session.query(DbGroupingRecord)
                .filter_by(hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
                .all()
            )
            assert len(matching_record) == 1

    def test_create_grouping_record_objects(self):
        """Tests create grouping record objects"""
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    hash=str(i) * 32,
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(2)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(2)],
        )
        records = grouping_lookup().create_grouping_record_objects(record_requests)

        assert len(records) == 2
        for i in range(len(record_requests.data)):
            assert records[i].hash == record_requests.data[i].hash
            assert records[i].project_id == record_requests.data[i].project_id
            assert records[i].message == record_requests.data[i].message
            assert records[i].stacktrace_embedding is not None

    def test_bulk_insert_new_grouping_records(self):
        """Test bulk inserting grouping records"""
        records = []
        hashes = [str(i) * 32 for i in range(10)]
        for i in range(10):
            embedding = grouping_lookup().encode_text("stacktrace " + str(i))
            new_record = GroupingRecord(
                hash=hashes[i],
                project_id=1,
                message="message " + str(i),
                stacktrace_embedding=embedding,
                group_id=None,
            ).to_db_model()
            records.append(new_record)

        grouping_lookup().bulk_insert_new_grouping_records(records)

        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            for i in range(10):
                assert records[i] is not None

    def test_bulk_create_and_insert_grouping_records_valid(self):
        """Test bulk creating and inserting grouping records"""
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=True)
        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            for i in range(10):
                assert records[i] is not None

    def test_delete_grouping_records_for_project(self):
        """Test deleting grouping records for a project"""
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=True)
        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            for i in range(10):
                assert records[i] is not None

        # Call the delete endpoint
        response = grouping_lookup().delete_grouping_records_for_project(1)

        # Verify records are deleted
        with Session() as session:
            records = (
                session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes)).all()
            )
            assert len(records) == 0

    def test_bulk_create_and_insert_grouping_records_invalid(self):
        """
        Test bulk creating and inserting grouping records fails when the input lists are of
        different lengths
        """
        hashes = [str(i) * 32 for i in range(10, 20)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(1)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=False)
        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            assert records.first() is None
