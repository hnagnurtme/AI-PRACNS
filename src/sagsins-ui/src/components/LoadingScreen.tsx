export default function LoadingScreen() {
  return (
    <div className="absolute inset-0 bg-black/60 z-10 flex flex-col items-center justify-center">
      <div className="relative mb-4">
        <div className="animate-spin rounded-full h-14 w-14 border-t-4 border-b-4 border-blue-400 border-opacity-70" />
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-white text-2xl">🚀</span>
        </div>
      </div>
      <h3 className="text-white text-lg font-semibold">Đang khởi tạo vệ tinh...</h3>
      <p className="text-white">Vui lòng chờ trong giây lát</p>
    </div>
  );
}
