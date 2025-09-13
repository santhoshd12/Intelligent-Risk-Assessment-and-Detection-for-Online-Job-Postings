import "./PostingTable.css";

const PostingTable = ({ data }) => {
  return (
    <table className="posting-table">
      <thead>
        <tr>
          <th>Date</th>
          <th>Platform</th>
          <th>Similarity (%)</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        {data.map((job, idx) => (
          <tr key={idx}>
            <td>{job.date}</td>
            <td>{job.platform}</td>
            <td>{job.similarity}</td>
            <td>{job.original}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default PostingTable;
