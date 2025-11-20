# Utah Mammoth NHL Analytics

A FastAPI backend and React TypeScript frontend application for querying and displaying Utah Mammoth NHL team information.

## Project Structure

```
NHLAnalytics/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── src/                # React source code
│   │   ├── components/     # React components
│   │   ├── services/       # API service
│   │   ├── types/          # TypeScript type definitions
│   │   ├── App.tsx         # Main App component
│   │   ├── App.css         # Styles
│   │   └── index.tsx       # Entry point
│   ├── public/             # Static files
│   ├── package.json        # Node.js dependencies
│   └── tsconfig.json       # TypeScript configuration
├── dataExploration/        # Jupyter notebook for data exploration
│   └── nhlAnalyticsV1.ipynb
├── start_react_app.bat     # Windows startup script
└── README.md
```

## Setup Instructions

### Backend Setup (FastAPI)

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:

   ```bash
   python main.py
   ```

   The backend will be available at `http://localhost:8000`

### Frontend Setup (React TypeScript)

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   ```

3. Run the React development server:

   ```bash
   npm start
   ```

   The frontend will be available at `http://localhost:3000` and will open automatically in your browser.

## API Endpoints

### Backend API (FastAPI)

- `GET /` - Root endpoint
- `GET /team/info` - Get basic team information
- `GET /team/stats` - Get team statistics
- `GET /team/roster` - Get team roster
- `GET /health` - Health check

### Frontend API (Express)

- `GET /` - Main dashboard
- `GET /api/team/info` - Proxy to backend team info
- `GET /api/team/stats` - Proxy to backend team stats
- `GET /api/team/roster` - Proxy to backend roster

## Features

- **Real-time Data**: Fetches live data from the NHL API
- **Responsive Design**: Modern, mobile-friendly interface
- **Team Statistics**: Displays wins, losses, points, goals, etc.
- **Player Roster**: Shows current team roster with stats
- **Auto-refresh**: Automatically refreshes data every 5 minutes
- **Error Handling**: Graceful error handling and user feedback

## About Utah Mammoth

The Utah Mammoth is a professional ice hockey team based in Salt Lake City, Utah, competing in the National Hockey League (NHL) as a member of the Central Division in the Western Conference. The team was established in 2024 when Ryan Smith, owner of the NBA's Utah Jazz, acquired the assets of the Arizona Coyotes and relocated the franchise to Utah.

## Technologies Used

- **Backend**: FastAPI, Python, Requests
- **Frontend**: React 18, TypeScript, Axios
- **Data Source**: NHL Official API (https://api.nhle.com/stats/rest)
- **Styling**: CSS3 with modern gradients and animations
- **Build Tool**: Create React App

## Development

### Quick Start (Windows)

Run the `start_react_app.bat` file to start both servers automatically.

### Manual Setup

To run both servers simultaneously:

1. Open two terminal windows
2. In the first terminal, start the backend:
   ```bash
   cd backend
   python main.py
   ```
3. In the second terminal, start the frontend:
   ```bash
   cd frontend
   npm start
   ```

Visit `http://localhost:3000` to see the application in action!

### Features

- **TypeScript**: Full type safety for API responses and components
- **Modern React**: Uses React 18 with functional components and hooks
- **Responsive Design**: Mobile-friendly interface with modern CSS
- **Real-time Data**: Fetches live data from the NHL API
- **Error Handling**: Graceful error handling and loading states
- **Auto-refresh**: Manual refresh button for updated data
