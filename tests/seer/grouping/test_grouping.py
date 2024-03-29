import time
import unittest
from sys import getsizeof

import pytest

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
    def test_create_grouping_record_objects(self):
        """Tests create grouping record objects"""
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
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
            assert records[i].group_id == record_requests.data[i].group_id
            assert records[i].project_id == record_requests.data[i].project_id
            assert records[i].message == record_requests.data[i].message
            assert records[i].stacktrace_embedding is not None

    def test_bulk_insert_new_grouping_records(self):
        """Test bulk inserting grouping records"""
        records = []
        for i in range(10):
            embedding = grouping_lookup().encode_text("stacktrace " + str(i))
            new_record = GroupingRecord(
                group_id=i,
                project_id=1,
                message="message " + str(i),
                stacktrace_embedding=embedding,
            ).to_db_model()
            records.append(new_record)

        grouping_lookup().bulk_insert_new_grouping_records(records)

        for i in range(10):
            with Session() as session:
                assert (
                    session.query(DbGroupingRecord).filter(DbGroupingRecord.group_id == i).first()
                    is not None
                )

    def test_bulk_create_and_insert_grouping_records_valid(self):
        """Test bulk creating and inserting grouping records"""
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(10)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=True)
        for i in range(10):
            with Session() as session:
                assert (
                    session.query(DbGroupingRecord).filter(DbGroupingRecord.group_id == i).first()
                    is not None
                )

    def test_bulk_create_and_insert_grouping_records_invalid(self):
        """
        Test bulk creating and inserting grouping records fails when the input lists are of
        different lengths
        """
        record_requests = CreateGroupingRecordsRequest(
            data=[
                CreateGroupingRecordData(
                    group_id=i,
                    project_id=1,
                    message="message " + str(i),
                )
                for i in range(10)
            ],
            stacktrace_list=["stacktrace " + str(i) for i in range(1)],
        )

        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_requests)
        assert response == BulkCreateGroupingRecordsResponse(success=False)
        for i in range(10):
            with Session() as session:
                assert (
                    session.query(DbGroupingRecord).filter(DbGroupingRecord.group_id == i).first()
                    is None
                )

    @pytest.mark.skip(reason="This test is purely for benchmarking how long this could take")
    def test_similarity_grouping_record_endpoint_2mb(self):
        """
        Test how long it will take to create and insert many records
        Aiming to test if 2mb is a good size
        """
        record_requests = []
        stacktrace_list = []
        stacktrace = (
            "SecurityError: Failed to read the 'cookie' property from 'Document': Cookies are disabled inside 'data:' URLs.\r\n  File \"/_static/dist/entrypoints/app-something.js\", line 1, in None\r\n    \r\n  File \"../../something/static/something/app/initializeCoolMetrics.tsx\", line 2, in initializeCoolMetrics\r\n    something.getEntriesByValue('value').forEach(value => {\r\n  File \"<anonymous>\", line None, in Array.forEach\r\n    \r\n  File \"../../something/static/something/app/initializeCoolMetrics.tsx\", line 227, in forEach\r\n    window.something?.metric(value.name, value.value, {"
            * 5
        )
        message = (
            "SecurityError: Failed to read the cookie property from Document: Cookies are disabled insude data URLs"
            * 5
        )
        i = 0
        while getsizeof(record_requests) < 1000:  # 2000000
            record_request = CreateGroupingRecordData(
                group_id=i,
                project_id=1,
                message=message + str(i),
            )
            record_requests.append(record_request)
            stacktrace_list.append(stacktrace + str(i))

        record_request_obj = CreateGroupingRecordsRequest(
            data=record_requests, stacktrace_list=stacktrace_list
        )
        start_time = time.time()
        response = grouping_lookup().bulk_create_and_insert_grouping_records(record_request_obj)
        end_time = time.time()

    def test_get_nearest_neighbors_no_hash_has_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has no stacktrace hash and a neighbor exists
        within the threshold.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1, project_id=1, stacktrace="stacktrace", message="message"
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            group_id=2, project_id=1, stacktrace=b"stacktrace", message="message"
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=1,
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
            token=None,
        )

    def test_get_nearest_neighbors_has_hash_has_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has a stacktrace hash and this hash matches
        an existing record (ie. the neighbor exists within the threshold)
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=1,
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
            token=None,
        )

    def test_get_nearest_neighbors_no_hash_no_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has no stacktrace hash and no neighbor exists
        within the threshold. Assert that a new record was added and its id is returned.
        """
        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = session.query(DbGroupingRecord).filter_by(group_id=2).first()

        assert new_record
        assert response == SimilarityResponse(responses=[], token=new_record.id)

    def test_get_nearest_neighbors_has_hash_no_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has a stacktrace hash and no neighbor exists
        within the threshold. Assert that a new record was added and its id is returned.
        """
        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = session.query(DbGroupingRecord).filter_by(group_id=2).first()

        assert new_record
        assert response == SimilarityResponse(responses=[], token=new_record.id)

    def test_get_nearest_neighbors_no_group_id(self):
        """
        Tests get_nearest_neighbors when the request has no group id and the matching record has
        no group id. Assert that a new record was added and its id is returned.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                project_id=1, stacktrace="stacktrace", message="message_1"
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            project_id=1, stacktrace=b"stacktrace", message="message", k=1, threshold=0.01
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = session.query(DbGroupingRecord).filter_by(message="message").first()

        assert new_record
        assert response == SimilarityResponse(responses=[], token=new_record.id)

    def test_insert_new_grouping_record_group_record_exists(self):
        """
        Tests that insert_new_grouping_record only creates one record per group id.
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                stacktrace_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            insert_id = grouping_lookup().insert_new_grouping_record(
                session, grouping_request, embedding
            )
            # Re-insert the grouping record
            existing_id = grouping_lookup().insert_new_grouping_record(
                session, grouping_request, embedding
            )
            assert insert_id == existing_id