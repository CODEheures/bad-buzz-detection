# Bad buzz detection

This ML application detect bad buzz.

## Modeling 
This is a web page which we can train model. If model is good we can ask to publish it to production
See it at http://16.171.224.228:8501

## Production 
This is a web page for end user. End user can predict tweet sentiment with this page

## CI/CD
This repo use Continuous Integrating for Machine learning
- Main Brach is protected. Merge request from non protected branch is required to commit modifications
- With Github workflow to run pipelines on push Merge request 
- Flake8 to Lint code and ensure PEP8 standard
- PyTest to unit test of modules and functions
- Test on production model to decide the merge (thanks https://cml.dev/doc to place reports with graphs on Pull Request comments)
