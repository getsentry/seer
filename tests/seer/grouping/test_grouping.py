import unittest

from johen import change_watcher
from johen.pytest import parametrize

from seer.db import DbGroupingRecord, Session
from seer.grouping.grouping import (
    BulkCreateGroupingRecordsResponse,
    CreateGroupingRecordData,
    CreateGroupingRecordsRequest,
    DeleteGroupingRecordsByHashRequest,
    DeleteGroupingRecordsByHashResponse,
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
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().upsert_grouping_record(embedding, grouping_request, session)
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

    def test_get_nearest_neighbors_no_neighbor_read_only(self):
        """
        Test read only get_nearest_neighbors when no matching record exists. Assert that the record
        for the group hash was not added.
        """
        grouping_request = GroupingRequest(
            hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            project_id=1,
            stacktrace="stacktrace",
            message="message",
            k=1,
            threshold=0.01,
            read_only=True,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = (
                session.query(DbGroupingRecord)
                .filter_by(hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
                .first()
            )

        assert new_record is None
        assert response == SimilarityResponse(responses=[])

    def test_insert_new_grouping_record_group_record_exists(self):
        """
        Tests that insert_new_grouping_record only creates one record per group hash.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            grouping_lookup().upsert_grouping_record(embedding, grouping_request, session)
            session.commit()
            # Re-insert the grouping record
            grouping_lookup().upsert_grouping_record(embedding, grouping_request, session)
            session.commit()
            matching_record = (
                session.query(DbGroupingRecord)
                .filter_by(hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
                .all()
            )
            assert len(matching_record) == 1

    def test_insert_new_grouping_record_group_record_cross_project(self):
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request1 = GroupingRequest(
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_request2 = GroupingRequest(
                project_id=2,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            grouping_lookup().upsert_grouping_record(embedding, grouping_request1, session)
            session.commit()
            grouping_lookup().upsert_grouping_record(embedding, grouping_request2, session)
            session.commit()
            matching_record = (
                session.query(DbGroupingRecord)
                .filter_by(hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
                .all()
            )
            assert len(matching_record) == 2

    def test_bulk_create_and_insert_grouping_records_valid(self):
        """Test bulk creating and inserting grouping records"""
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=True, groups_with_neighbor={})
        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            for i in range(10):
                assert records[i] is not None

    def test_bulk_create_and_insert_grouping_records_with_existing_record(self):
        """Test bulk creating and inserting grouping records with one record already existing"""
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        # Insert one record before bulk insert
        single_record = GroupingRecord(
            hash=hashes[0],
            project_id=1,
            message="message " + str(0),
            stacktrace_embedding=grouping_lookup().encode_text("stacktrace " + str(0)),
        ).to_db_model()
        with Session() as session:
            session.add(single_record)
            session.commit()

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response.success is True
        assert len(response.groups_with_neighbor) == 0

        with Session() as session:
            assert session.query(DbGroupingRecord).filter(
                DbGroupingRecord.hash.in_(hashes)
            ).count() == len(hashes)

    def test_delete_grouping_records_by_hash(self):
        """Test deleting grouping records by hash list"""
        # Create 10 records
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)

        # Call the delete endpoint to delete first 5 records
        delete_hashes = hashes[:5]
        request_data = DeleteGroupingRecordsByHashRequest(project_id=1, hash_list=delete_hashes)
        response = grouping_lookup().delete_grouping_records_by_hash(request_data)
        assert response == DeleteGroupingRecordsByHashResponse(success=True)

        # Verify only delete hashes are deleted
        with Session() as session:
            records = (
                session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes)).all()
            )
            assert len(records) == 5
            for hash in delete_hashes:
                assert hash not in records

    def test_bulk_create_and_insert_grouping_records_invalid(self):
        """
        Test bulk creating and inserting grouping records fails when the input lists are of
        different lengths
        """
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(1)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=False, groups_with_neighbor={})
        with Session() as session:
            records = session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes))
            assert records.first() is None

    def test_bulk_create_and_insert_grouping_records_has_neighbor_in_existing_records(self):
        """
        Test bulk creating and inserting grouping records does not create a record for a hash that
        has a nearest neighbor that exists in existing records.
        """
        # Create a record with the stacktrace "stacktrace"
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().upsert_grouping_record(embedding, grouping_request, session)
            session.commit()

        # Create record data to attempt to be inserted, create 5 with the stacktrace "stacktrace"
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message",
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace" for _ in range(5)]
            + ["something different " + str(i) for i in range(6, 11)],
        )

        expected_groups_with_neighbor = {}
        for i in range(5):
            expected_groups_with_neighbor[str(i)] = GroupingResponse(
                parent_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
                stacktrace_distance=0.00,
                message_distance=0.00,
                should_group=True,
            )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(
            success=True, groups_with_neighbor=expected_groups_with_neighbor
        )
        with Session() as session:
            records_without_neighbor = (
                session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes[5:])).all()
            )
            assert len(records_without_neighbor) == 5
            records_with_neighbor = session.query(DbGroupingRecord).filter(
                DbGroupingRecord.hash.in_(hashes[:5])
            )
            assert records_with_neighbor.all() == []

    def test_bulk_create_and_insert_grouping_records_has_neighbor_in_batch(self):
        """
        Test bulk creating and inserting grouping records does not create a record for a hash that
        has a nearest neighbor that exists in the current batch.
        """
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message",
                )
                for i in range(10)
            ],
            # Create 5 duplicate stacktraces
            stacktrace_list=["stacktrace " + str(i) for i in range(5)]
            + ["stacktrace " + str(i) for i in range(5)],
        )

        # We expect the last 5 entries to have a neighbor
        expected_groups_with_neighbor = {}
        for i in range(5):
            expected_groups_with_neighbor[str(i + 5)] = GroupingResponse(
                parent_hash=str(i) * 32,
                stacktrace_distance=0.00,
                message_distance=0.00,
                should_group=True,
            )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(
            success=True, groups_with_neighbor=expected_groups_with_neighbor
        )
        with Session() as session:
            records = (
                session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes)).all()
            )
            assert len(records) == 5
            no_records = (
                session.query(DbGroupingRecord).filter(DbGroupingRecord.hash.in_(hashes[5:])).all()
            )
            assert len(no_records) == 0

    def test_delete_grouping_records_for_project(self):
        """Test deleting grouping records for a project"""
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=True, groups_with_neighbor={})
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


@parametrize(count=1)
def test_upsert_grouping_record(
    project_1_id: int,
    hash_1: str,
    orig_record: GroupingRequest,
):
    orig_record.project_id = project_1_id
    orig_record.hash = hash_1
    embedding = grouping_lookup().encode_text("some text")
    test_records = [
        orig_record,
        orig_record.copy(update=dict(project_id=(project_1_id + 1))),
        orig_record.copy(update=dict(hash=(hash_1 + "_2"))),
        # Duplicate of original should not actually update the original
        orig_record.copy(update=dict(message=orig_record.message + " updated?")),
    ]

    def query_created(record: GroupingRequest) -> DbGroupingRecord | None:
        with Session() as session:
            return (
                session.query(DbGroupingRecord)
                .filter_by(hash=record.hash, project_id=record.project_id)
                .first()
            )

    @change_watcher
    def updated_message_for_orig():
        db_record = query_created(orig_record)
        if db_record:
            return db_record.message
        return None

    duplicates = 0
    with updated_message_for_orig as changed:
        with Session() as session:
            for record in test_records:
                duplicates += int(
                    grouping_lookup().upsert_grouping_record(embedding, record, session)
                )
            session.commit()

    assert duplicates == 1
    assert changed
    assert changed.to_value(orig_record.message)

    # ensure that atleast a record was made for each item
    for item in test_records:
        assert query_created(item) is not None
