"""Run the NumericBridge contribution MCP over local stdio."""

import argparse
import asyncio
from pathlib import Path
from typing import Annotated, Any

from mcp import types
from mcp.server.mcpserver import Context, MCPServer
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field, StrictBool, StringConstraints

from .intake import IntakeClient
from .records.submission import Submissions


ROOT = Path(__file__).resolve().parents[1]
Title = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1,
                                        max_length=120, pattern=r'^[^\x00-\x1f\x7f]+$')]
DraftID = Annotated[str, StringConstraints(strict=True, pattern=r'^draft-[a-f0-9]{64}$')]
DraftHash = Annotated[str, StringConstraints(strict=True, pattern=r'^[a-f0-9]{64}$')]
Record = dict[str, Any]
Records = Annotated[list[Record], Field(max_length=50)]
Evidence = Annotated[dict[str, Record], Field(max_length=20)]


class Approval(BaseModel):
    approve: StrictBool = Field(description='Send this exact reviewed draft to NumericBridge for community review')


def create_server(submissions: Submissions, intake: IntakeClient | None = None) -> MCPServer:
    server = MCPServer('numericbridge-contribution', version='1.0.0', subscriptions=False,
                       instructions='Show the local preview, then use submit_community for the only sharing confirmation. Never answer it on the user\'s behalf.')

    @server.tool(
        title='Prepare a NumericBridge contribution',
        annotations=types.ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    async def prepare_community_submission(
        title: Title,
        evaluations: Records | None = None,
        repairs: Records | None = None,
        library_groups: Records | None = None,
        entities: Records | None = None,
        links: Records | None = None,
        evidence: Evidence | None = None,
    ) -> dict[str, Any]:
        """Validate additions and save an exact local preview without sending it.

        Only include public, reviewed summaries and minimum evidence. Evidence
        maps a new site-relative evidence/submissions/*.json path to its JSON
        object. Reuse the community record schemas for all supplied records.
        Show the actual additions and destination, then offer submission through
        the client confirmation.
        """
        additions = dict(evaluations=dict(evaluations=evaluations or []),
                         repairs=dict(repairs=repairs or [], library_groups=library_groups or []),
                         relations=dict(entities=entities or [], links=links or []), evidence=evidence or {})
        try:
            return await asyncio.to_thread(submissions.prepare, title, additions)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RecursionError) as error:
            return dict(status='invalid', findings=[str(error)])

    @server.tool(
        title='Submit a reviewed NumericBridge contribution',
        annotations=types.ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    async def submit_community(draft_id: DraftID, draft_sha256: DraftHash, ctx: Context) -> dict[str, Any]:
        """Request confirmation and send this exact draft to the community inbox.

        After showing the preview, use this tool as the sharing offer. The client
        presents the only user confirmation. Rejecting or cancelling sends nothing.
        """
        try:
            draft = await asyncio.to_thread(submissions.reviewed, draft_id, draft_sha256)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(status='invalid_draft', message=str(error))
        if intake is None:
            return dict(status='intake_unavailable', message='No community inbox is configured; keep this local draft')
        session = ctx.request_context.session
        if not session.check_client_capability(types.ClientCapabilities(elicitation=types.ElicitationCapability())):
            return dict(status='approval_unavailable', message='Use a client that presents MCP form elicitation to the user')
        try:
            approval = await ctx.elicit(
                f"Share this anonymized result with NumericBridge so others can reuse it?\n"
                f"Title: {draft['title']}\nCommunity inbox: {draft['destination']}\n"
                f"Draft SHA-256: {draft_sha256}", Approval)
        except (MCPError, TypeError, ValueError):
            return dict(status='approval_unavailable', message='The client could not confirm this exact draft')
        if approval.action != 'accept' or approval.data.approve is not True:
            return dict(status='declined' if approval.action != 'cancel' else 'cancelled')
        try:
            return await asyncio.to_thread(intake.send, draft, draft_sha256)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(status='submission_failed', message=str(error))

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', help='community submission URL; omit for local drafts only')
    parser.add_argument('--token-file', type=Path, help='file containing the community upload credential')
    parser.add_argument('--drafts', type=Path, default=Path.home() / '.numericbridge' / 'drafts',
                        help='private local draft directory')
    args = parser.parse_args()
    if bool(args.intake) != bool(args.token_file):
        parser.error('--intake and --token-file must be configured together')
    try:
        intake = IntakeClient(args.intake, args.token_file) if args.intake else None
    except (OSError, ValueError) as error:
        parser.error(str(error))
    submissions = Submissions(ROOT, args.drafts, destination=intake.url if intake else None)
    create_server(submissions, intake).run('stdio')


if __name__ == '__main__':
    main()
