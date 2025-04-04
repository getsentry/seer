import unittest
import uuid
from unittest import mock

import numpy as np
import torch
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
    _load_model,
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
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            project_id=1,
            stacktrace="stacktrace",
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
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            grouping_lookup().insert_new_grouping_record(grouping_request, embedding)
            # Re-insert the grouping record
            grouping_lookup().insert_new_grouping_record(grouping_request, embedding)
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
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_request2 = GroupingRequest(
                project_id=2,
                stacktrace="stacktrace",
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            grouping_lookup().insert_new_grouping_record(grouping_request1, embedding)
            grouping_lookup().insert_new_grouping_record(grouping_request2, embedding)
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
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        # Insert one record before bulk insert
        single_record = GroupingRecord(
            hash=hashes[0],
            project_id=1,
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
                hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(grouping_request, embedding)

        # Create record data to attempt to be inserted, create 5 with the stacktrace "stacktrace"
        hashes = [str(i) * 32 for i in range(10)]
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    hash=hashes[i],
                    project_id=1,
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
                should_group=True,
            )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        for group_response in response.groups_with_neighbor.values():
            group_response.stacktrace_distance = round(group_response.stacktrace_distance, 3)
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

    def test_rerank_candidates(self):
        """
        Test that the rerank_candidates method correctly reranks candidates based on cosine distance.
        """
        # Create mock candidates with incorrect initial ordering
        embedding = np.array([1, 2, 3], dtype=np.float32)
        candidates = [
            DbGroupingRecord(
                stacktrace_embedding=np.array([7, 8, 9], dtype=np.float32), hash="a" * 32
            ),
            DbGroupingRecord(
                stacktrace_embedding=np.array([4, 5, 6], dtype=np.float32), hash="b" * 32
            ),
            DbGroupingRecord(
                stacktrace_embedding=np.array([1, 2, 3], dtype=np.float32), hash="c" * 32
            ),
            DbGroupingRecord(
                stacktrace_embedding=np.array([-1, -2, -3], dtype=np.float32), hash="d" * 32
            ),
        ]

        # Set a distance threshold that includes all candidates except for the one that is reversed
        distance_threshold = 0.05

        # Rerank candidates
        reranked = grouping_lookup().rerank_candidates(
            candidates, embedding, distance_threshold, "abcdef"
        )

        # Check that the order is correct after reranking
        self.assertEqual(len(reranked), 3)
        self.assertAlmostEqual(reranked[0][1], 0.0, places=6)  # Should be exact match
        self.assertAlmostEqual(reranked[1][1], 0.0253682, places=6)
        self.assertAlmostEqual(reranked[2][1], 0.0405881, places=6)

        # Check that the order of candidates is corrected
        self.assertEqual(reranked[0][0], candidates[2])  # [1, 2, 3] should be first now
        self.assertEqual(reranked[1][0], candidates[1])  # [4, 5, 6] should be second
        self.assertEqual(reranked[2][0], candidates[0])  # [7, 8, 9] should be 3rd

        # Verify that the initial order was incorrect
        self.assertNotEqual(candidates[0], reranked[0][0])
        self.assertNotEqual(candidates[2], reranked[2][0])
        
    def test_handle_device_id_error(self):
        """
        Test that the handle_out_of_memory decorator catches device ID errors.
        """
        from seer.grouping.grouping import handle_out_of_memory
        
        # Create a function that raises a RuntimeError with 'device ID' in the message
        @handle_out_of_memory
        def function_with_device_id_error():
            raise RuntimeError("invalid device ID 999")
            
        # The function should not raise an exception because the decorator should catch it
        function_with_device_id_error()  # Should not raise
        
    @mock.patch('torch.device')
    @mock.patch('sentry_sdk.logging.logger.warning')
    @mock.patch('sentence_transformers.SentenceTransformer')
    def test_load_model_fallback_to_cpu(self, mock_transformer, mock_warning, mock_device):
        """
        Test that _load_model falls back to CPU when a device ID error is raised.
        """
        # Setup the mock to raise a RuntimeError with 'device ID' when first called
        # and return a valid model on the second call
        mock_transformer.side_effect = [
            RuntimeError("invalid device ID 999"),
            mock.MagicMock()
        ]
        
        # Setup device mock to return "cuda" for the first call
        mock_device.side_effect = lambda device: mock.MagicMock(spec=torch.device, __str__=lambda _: device)
        
        # Call _load_model, which should handle the error and retry with CPU
        model = _load_model("test_model_path")
        
        # Verify the model was loaded correctly
        self.assertIsNotNone(model)
        
        # Verify _load_model was called twice, first with cuda then with cpu
        self.assertEqual(mock_transformer.call_count, 2)
        
        # Verify the warning was logged
        mock_warning.assert_called_once()


@parametrize(count=1)
def test_GroupingLookup_insert_batch_grouping_records_duplicates(
    project_1_id: int,
    hash_1: str,
    orig_record: CreateGroupingRecordData,
    grouping_request: CreateGroupingRecordsRequest,
):
    orig_record.project_id = project_1_id
    orig_record.hash = hash_1
    orig_record.exception_type = "error"
    project_2_id = project_1_id + 1
    hash_2 = hash_1 + "_2"

    updated_duplicate = orig_record.copy(update=dict(exception_type="transaction"))

    grouping_request.data = [
        orig_record,
        orig_record.copy(update=dict(project_id=project_2_id)),
        orig_record.copy(update=dict(hash=hash_2)),
        # Duplicate of original should not actually update the original
        updated_duplicate,
    ]
    grouping_request.stacktrace_list = [uuid.uuid4().hex for r in grouping_request.data]

    def query_created(record: CreateGroupingRecordData) -> DbGroupingRecord | None:
        with Session() as session:
            return (
                session.query(DbGroupingRecord)
                .filter_by(hash=record.hash, project_id=record.project_id)
                .first()
            )

    @change_watcher
    def updated_exception_type_for_orig():
        db_record = query_created(orig_record)
        if db_record:
            return db_record.error_type
        return None

    with updated_exception_type_for_orig as changed:
        grouping_lookup().insert_batch_grouping_records(grouping_request)

    assert changed
    assert changed.to_value(orig_record.exception_type)

    # ensure that atleast a record was made for each item
    for item in grouping_request.data:
        assert query_created(item) is not None

    # Again, ensuring that duplicates are ignored
    grouping_request.data = [updated_duplicate]
    grouping_request.stacktrace_list = ["does not matter" for _ in grouping_request.data]
    with updated_exception_type_for_orig as changed:
        grouping_lookup().insert_batch_grouping_records(grouping_request)

    assert not changed
