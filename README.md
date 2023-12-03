# Bad buzz detection

This applicaion detect bad buzz.

## Modeling 
This is a web page which we can train model. If model is good we can ask to publish it to production

## Production 
This is a web page for end user. End user can predict tweet sentiment with this page

## CI/CD
This repo use Continuous Integrating for Machine learning
- With Github workflow to run pipelines on push Merge request 
- Flake8 to Lint code and ensure PEP8 standard
- PyTest to unit test of modules and functions
- Test on production model to decide the merge (thanks https://cml.dev/doc to place reports with graphs on Pull Request comments)
