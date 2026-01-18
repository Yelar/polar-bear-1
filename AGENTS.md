# Repository Guidelines

## Project Structure & Module Organization
- `polar-bear/` contains the macOS app source and Xcode project.
  - `polar-bear/polar-bear.xcodeproj` is the Xcode project.
  - `polar-bear/polar-bear/` holds Swift sources and assets.
- `backend/` contains the FastAPI service that calls Tokenc in Python.
  - `backend/main.py` is the API entrypoint.
  - `backend/requirements.txt` lists Python dependencies.
  - `backend/.env` (user-created) stores API keys.

## Build, Test, and Development Commands
- macOS app (Xcode):
  - Open `polar-bear/polar-bear.xcodeproj` and run the app target from Xcode.
  - Menu bar app only; settings open from the status item.
- Backend:
  - Create and activate a venv:
    - `cd backend`
    - `python -m venv .venv`
    - `source .venv/bin/activate`
  - Install deps: `pip install -r requirements.txt`
  - Run server: `uvicorn main:app --reload`

## Coding Style & Naming Conventions
- Swift: 4-space indentation, UpperCamelCase for types, lowerCamelCase for methods and vars.
- Python: 4-space indentation, snake_case for functions and vars, UpperCamelCase for classes.
- Keep files focused by responsibility (e.g., `AccessibilityService.swift`, `NetworkClient.swift`).

## Testing Guidelines
- No automated tests are configured yet for either the app or backend.
- When adding tests, keep them close to their module (e.g., `backend/tests/`).

## Commit & Pull Request Guidelines
- No commit message convention is established in the repository history.
- Recommended: short, imperative summaries (e.g., "Add FastAPI backend for tokenc").
- PRs should include:
  - A concise description of changes.
  - Steps to run or verify (e.g., how to start `uvicorn`).
  - Screenshots or recordings for UI changes.

## Security & Configuration Tips
- Store keys in `backend/.env` only. Do not commit real secrets.
- The macOS app reads from the local backend URL; keep it bound to `127.0.0.1` for dev.
