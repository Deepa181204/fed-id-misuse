// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;


import React, { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:5000"; // Flask API

function App(){
  const [clients, setClients] = useState([]);
  const [updates, setUpdates] = useState([]);
  const [selected, setSelected] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [topk, setTopk] = useState(50);
  const [showFeatures, setShowFeatures] = useState(false);

  useEffect(() => {
    fetchClients();
    fetchUpdates();
    const id = setInterval(fetchUpdates, 5000); // refresh updates
    return () => clearInterval(id);
  }, []);

  function fetchClients(){
    fetch(`${API_BASE}/clients`).then(r=>r.json()).then(setClients).catch(e=>console.error(e));
  }
  function fetchUpdates(){
    fetch(`${API_BASE}/updates?limit=50`).then(r=>r.json()).then(setUpdates).catch(e=>console.error(e));
  }

  function selectClient(client){
    setSelected(client);
    setAnomalies([]);
    setLoading(true);
    const params = new URLSearchParams({
      client_id: client,
      top: topk,
      return_features: showFeatures ? "1" : "0"
    });
    fetch(`${API_BASE}/anomalies?${params.toString()}`).then(r=>{
      if(!r.ok) throw new Error("failed");
      return r.json();
    }).then(data=>{
      setAnomalies(data.results || []);
    }).catch(err=>{
      console.error(err);
      setAnomalies([]);
    }).finally(()=>setLoading(false));
  }

  return (
    <div style={{fontFamily: "Arial, sans-serif", padding: 20}}>
      <h1>Fed-ID-Misuse Dashboard</h1>

      <section style={{display: "flex", gap: 20}}>
        <div style={{flex: 1}}>
          <h3>Clients</h3>
          <ul>
            {clients.map(c=>(
              <li key={c}>
                <button onClick={()=>selectClient(c)} style={{marginRight:8}}>{c}</button>
              </li>
            ))}
          </ul>
        </div>

        <div style={{flex: 2}}>
          <h3>Recent Updates</h3>
          <table border="1" cellPadding="6">
            <thead><tr><th>ID</th><th>Client</th><th>Accuracy</th><th>TS</th></tr></thead>
            <tbody>
              {updates.map(u=>(
                <tr key={u.id}>
                  <td>{u.id}</td>
                  <td>{u.client_id}</td>
                  <td>{u.accuracy?.toFixed?.(4)}</td>
                  <td>{u.ts}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <hr/>

      <section>
        <h3>Anomalies {selected ? `for ${selected}` : ""}</h3>
        <div style={{marginBottom:8}}>
          <label>Top K: <input type="number" value={topk} onChange={e=>setTopk(Number(e.target.value))} style={{width:80}}/></label>
          <label style={{marginLeft:12}}>
            <input type="checkbox" checked={showFeatures} onChange={e=>setShowFeatures(e.target.checked)} /> Show features
          </label>
          <button style={{marginLeft:12}} onClick={()=>selected && selectClient(selected)}>Refresh</button>
        </div>

        {loading && <div>Loading anomalies...</div>}

        {!loading && anomalies.length === 0 && selected && <div>No anomalies found or no data.</div>}

        {!loading && anomalies.length>0 && (
          <table border="1" cellPadding="6" style={{width:"100%", borderCollapse:"collapse"}}>
            <thead>
              <tr><th>Idx</th><th>MSE</th><th>Anomaly?</th>{showFeatures && <th>Features</th>}</tr>
            </thead>
            <tbody>
              {anomalies.map(row=>(
                <tr key={row.idx}>
                  <td>{row.idx}</td>
                  <td>{row.mse.toFixed(6)}</td>
                  <td>{row.is_anomaly ? "✅" : "—"}</td>
                  {showFeatures && <td><pre style={{whiteSpace:"pre-wrap"}}>{JSON.stringify(row.features)}</pre></td>}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}

export default App;
