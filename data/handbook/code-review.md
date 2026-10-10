# Code Review Guidelines

Code review is how we share knowledge and keep the fleet safe. These guidelines apply to every repository.

## Approvals

Most changes need one approval from a code owner listed in the repository's CODEOWNERS file. Changes under the `critical/` directory, which contains motion control and safety logic, need two approvals, and at least one of them must come from the Fleet Safety team.

You may not approve your own pull request, even if you are an administrator of the repository.

## Pull request size

Keep pull requests under 400 changed lines where possible. Large changes are harder to review and more likely to hide bugs. If a change has to be large, split it into a stack of smaller pull requests or walk a reviewer through it on a call first.

## Response times

Reviewers should give a first response within one business day. If you cannot review in time, say so and suggest another reviewer. Authors should not merge a change that has unresolved comments marked as blocking.

## Writing review comments

Prefix comments to make their weight clear:

- `nit:` for small style issues that should not block a merge.
- `question:` when you want to understand the change, not request a change.
- `blocking:` when the change must not merge until the comment is addressed.

Review the code, not the author. Explain why you suggest a change.

## Automated checks

Every pull request must pass linting, unit tests, and the static analysis job before it can merge. Test coverage for new code in `critical/` must be at least 90%.
