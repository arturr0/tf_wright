// script.js
async function showChart(data) {
  // Set up a visor to visualize the data
  const surface = tfvis.visor().surface({ name: 'My Data', tab: 'Charts' });

  // Render a bar chart using tfvis
  const chartData = {
    values: data.arraySync(),
    labels: ['1', '2', '3', '4', '5']
  };

  tfvis.render.barchart(surface, chartData, {});
}

// Ensure loadData is available before calling it
loadData().then(data => {
  showChart(data);
});
