# Bad buzz detection

This ML application detect bad buzz.

## Modeling 
This is a web page which we can train model. If model is good we can ask to publish it to production
See it at https://train.air-paradis.codeheures.fr

## MlFlow 
This is a web page which we can validate model for production.
See it at https://mlflow.air-paradis.codeheures.fr

## Production 
This is a web page for end user. End user can predict tweet sentiment with this page.
See it at https://predict.air-paradis.codeheures.fr

## CI/CD
This repo use Continuous Integrating for Machine learning
- Main Brach is protected. Merge request from non protected branch is required to commit modifications
- With Github workflow to run pipelines on push Merge request 
- Flake8 to Lint code and ensure PEP8 standard
- PyTest to unit test of modules and functions
- Test on production model to decide the merge (thanks https://cml.dev/doc to place reports with graphs on Pull Request comments)
