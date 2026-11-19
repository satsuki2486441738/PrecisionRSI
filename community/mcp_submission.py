"""Run the NumericBridge contribution MCP over local stdio."""

import argparse
import asyncio
from pathlib import Path
from typing import Annotated, Any

from mcp import types
from mcp.server.mcpserver import Context, MCPServer
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, Field, StrictBool, StringConstraints

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
    approve: StrictBool = Field(description='Approve publishing this exact reviewed draft as a PR')


def create_server(submissions: Submissions) -> MCPServer:
    server = MCPServer('numericbridge-contribution', version='1.0.0', subscriptions=False,
                       instructions='Prepare a local contribution preview. Show it to the user before requesting approval; never answer an approval prompt on their behalf.')

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
        """Validate additions and save an exact local preview. No data goes to GitHub.

        Only include public, reviewed summaries and minimum evidence. Evidence
        maps a new site-relative evidence/submissions/*.json path to its JSON
        object. Reuse the community record schemas for all supplied records.
        Show the actual additions and destination to the user before submission.
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
        """Request the user's confirmation and submit this exact draft as a GitHub PR.

        The client must present the confirmation to the user. The call has no
        approval argument. Rejecting or cancelling leaves GitHub unchanged.
        """
        try:
            draft = await asyncio.to_thread(submissions.reviewed, draft_id, draft_sha256)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(status='invalid_draft', message=str(error))
        session = ctx.request_context.session
        if not session.check_client_capability(types.ClientCapabilities(elicitation=types.ElicitationCapability())):
            return dict(status='approval_unavailable', message='Use a client that presents MCP form elicitation to the user')
        try:
            approval = await ctx.elicit(
                f"Publish the reviewed NumericBridge draft as a pull request?\n"
                f"Title: {draft['title']}\nDestination: {draft['upstream']}:{draft['base_branch']}\n"
                f"From: {draft['fork']}\nDraft SHA-256: {draft_sha256}", Approval)
        except (MCPError, TypeError, ValueError):
            return dict(status='approval_unavailable', message='The client could not confirm this exact draft')
        if approval.action != 'accept' or approval.data.approve is not True:
            return dict(status='declined' if approval.action != 'cancel' else 'cancelled')
        try:
            return await asyncio.to_thread(submissions.publish, draft_id, draft_sha256)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return dict(status='submission_failed', message=str(error))

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork', required=True, help='contributor-owned GitHub fork OWNER/REPO')
    parser.add_argument('--drafts', type=Path, default=Path.home() / '.numericbridge' / 'drafts',
                        help='private local draft directory')
    args = parser.parse_args()
    try:
        submissions = Submissions(ROOT, args.drafts, args.fork)
    except ValueError as error:
        parser.error(str(error))
    create_server(submissions).run('stdio')


if __name__ == '__main__':
    main()
