import unittest

from seer.db import DbGroupingRecord, Session
from seer.grouping.grouping import GroupingRequest, GroupingResponse, SimilarityResponse
from seer.inference_models import grouping_lookup


class TestGrouping(unittest.TestCase):
    def test_get_nearest_neighbors_has_neighbor(self):
        """
        Tests get_nearest_neighbors when the request has a group hash and this hash matches
        an existing record (ie. the neighbor exists within the threshold)
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                group_id=1,
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            group_id=2,
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            group_hash="13501807435378261861369456856144",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=1,
                    parent_group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
                    stacktrace_distance=0.0,
                    message_distance=0.0,
                    should_group=True,
                )
            ],
        )

    def test_get_nearest_neighbors_has_neighbor_no_group_id(self):
        """
        Tests get_nearest_neighbors when the request has a group hash and this hash matches
        an existing record (ie. the neighbor exists within the threshold)
        """
        with Session() as session:
            embedding = grouping_lookup().encode_text("stacktrace")
            grouping_request = GroupingRequest(
                project_id=1,
                stacktrace="stacktrace",
                message="message",
                group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()

        grouping_request = GroupingRequest(
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            group_hash="13501807435378261861369456856144",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        assert response == SimilarityResponse(
            responses=[
                GroupingResponse(
                    parent_group_id=None,
                    parent_group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
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
            group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            project_id=1,
            stacktrace=b"stacktrace",
            message="message",
            k=1,
            threshold=0.01,
        )

        response = grouping_lookup().get_nearest_neighbors(grouping_request)
        with Session() as session:
            new_record = (
                session.query(DbGroupingRecord)
                .filter_by(group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
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
                group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD",
            )
            # Insert the grouping record
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()
            # Re-insert the grouping record
            grouping_lookup().insert_new_grouping_record(session, grouping_request, embedding)
            session.commit()
            new_record = (
                session.query(DbGroupingRecord)
                .filter_by(group_hash="QYK7aNYNnp5FgSev9Np1soqb1SdtyahD")
                .all()
            )
            assert len(new_record) == 1
