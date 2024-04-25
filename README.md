# Numerical WebSnapse (Server)

This is a project for visualizing and simulating **Numerical Spiking Neural P Systems**. The live application can be accessed [here](https://numerical-websnapse.vercel.app).

To clone the project:

```bash
git clone https://github.com/numerical-websnapse/numerical-websnapse-server
cd numerical-websnapse-server
```

*Note that you will need to have [Python 3.9](https://www.python.org)  or above installed.*

## Get started

Install the dependencies:

```bash
cd numerical-websnapse-server
pip install -r requirements.txt
```

then start [Uvicorn](https://www.uvicorn.org):

```bash
uvicorn app.socket:app --reload
```

## Test Cases

This repository provides test case generation for the full-stack application. To access this, first create a folder named `tests` under the `app` folder. Then in the `model` folder create a `generator` for your test system that routes the generated files to `test/<generator>`. Then run the generator:

```bash
python app/model/generators/<generator>.py
```

You can try out generating using `subsetsum.py` example.

## Deploying to the web

### With [Render](https://render.com)

Create a render profile if you haven't already. Go to your dashboard then create a new web service. choose the option:

> Build and deploy from a Git repository

Make sure to create a new repository under your profile and set the branch to `master`. Set the build command as:

```
pip install -r requirements.txt
```

Set the start command as:

```
python main.py
```

Select your instance type then create your web service.

## Development

The file structure for the server-side of Numerical WebSnapse is shown as:

```
app/
├─ middleware/
│  ├─ nsnp_validation.py
├─ model/
│  ├─ converter/
│  ├─ generators/
│  ├─ NSNP.py
├─ tests/
├─ socket.py
convert.py
main.py
requirements.txt
```

The model simulation logic is under `NSNP.py` where its implementation follows the matrix representation for NSN P systems [[1]](https://link.springer.com/article/10.1007/s41965-022-00093-7). Validation and type conversions are handled by `nsnp_validation.py` and the server **API** definitions are under `socket.py`.

When working together with the [client implementation](https://github.com/numerical-websnapse/numerical-websnapse-client), be sure to check where your local instance is hosted to set the system settings for the client accordingly. By default the local server url is defined as:

```
url: '127.0.0.1:8000'
```

## Author

Rey Christian E. Delos Reyes
