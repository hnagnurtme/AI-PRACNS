interface ErrorScreenProps {
  error: string;
  onRetry?: () => void;
}

export default function ErrorScreen({ error, onRetry }: ErrorScreenProps) {
  return (
    <div className="text-red-500 p-8 flex flex-col items-center justify-center">
      <h3 className="font-bold text-xl mb-2">❌ Có lỗi xảy ra</h3>
      <p className="mb-4">{error}</p>
      {onRetry && (
        <button
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition"
          onClick={onRetry}
        >
          Thử lại
        </button>
      )}
    </div>
  );
}
