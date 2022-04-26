/*******************************************************************************
 * Base Consumer thread:
 *   Creates an EGLStream::FrameConsumer object to read frames from the
 *   OutputStream, then creates/populates an NvBuffer (dmabuf) from the frames
 *   to be processed by processV4L2Fd.
 ******************************************************************************/
class ConsumerThread : public Thread
{
public:
    explicit ConsumerThread(OutputStream* stream) :
        m_stream(stream),
        m_dmabuf(-1)
    {
    }
    virtual ~ConsumerThread();

protected:
    /** @name Thread methods */
    /**@{*/
    virtual bool threadInitialize();
    virtual bool threadExecute();
    virtual bool threadShutdown();
    /**@}*/

    virtual bool processV4L2Fd(int32_t fd, uint64_t frameNumber) = 0;

    OutputStream* m_stream;
    UniqueObj<FrameConsumer> m_consumer;
    int m_dmabuf;
};

ConsumerThread::~ConsumerThread()
{
    if (m_dmabuf != -1)
        NvBufferDestroy(m_dmabuf);
}

bool ConsumerThread::threadInitialize()
{
    /* Create the FrameConsumer. */
    m_consumer = UniqueObj<FrameConsumer>(FrameConsumer::create(m_stream));
    if (!m_consumer)
        ORIGINATE_ERROR("Failed to create FrameConsumer");

    return true;
}

bool ConsumerThread::threadExecute()
{
    IEGLOutputStream *iEglOutputStream = interface_cast<IEGLOutputStream>(m_stream);
    IFrameConsumer *iFrameConsumer = interface_cast<IFrameConsumer>(m_consumer);

    /* Wait until the producer has connected to the stream. */
    CONSUMER_PRINT("Waiting until producer is connected...\n");
    if (iEglOutputStream->waitUntilConnected() != STATUS_OK)
        ORIGINATE_ERROR("Stream failed to connect.");
    CONSUMER_PRINT("Producer has connected; continuing.\n");

    while (true)
    {
        /* Acquire a frame. */
        UniqueObj<Frame> frame(iFrameConsumer->acquireFrame());
        IFrame *iFrame = interface_cast<IFrame>(frame);
        if (!iFrame)
            break;

        /* Get the IImageNativeBuffer extension interface. */
        NV::IImageNativeBuffer *iNativeBuffer =
            interface_cast<NV::IImageNativeBuffer>(iFrame->getImage());
        if (!iNativeBuffer)
            ORIGINATE_ERROR("IImageNativeBuffer not supported by Image.");

        /* If we don't already have a buffer, create one from this image.
           Otherwise, just blit to our buffer. */
        if (m_dmabuf == -1)
        {
            m_dmabuf = iNativeBuffer->createNvBuffer(iEglOutputStream->getResolution(),
                                                     NvBufferColorFormat_YUV420,
                                                     NvBufferLayout_BlockLinear);
            if (m_dmabuf == -1)
                CONSUMER_PRINT("\tFailed to create NvBuffer\n");
        }
        else if (iNativeBuffer->copyToNvBuffer(m_dmabuf) != STATUS_OK)
        {
            ORIGINATE_ERROR("Failed to copy frame to NvBuffer.");
        }

        /* Process frame. */
        processV4L2Fd(m_dmabuf, iFrame->getNumber());
    }

    CONSUMER_PRINT("Done.\n");

    requestShutdown();

    return true;
}

bool ConsumerThread::threadShutdown()
{
    return true;
}

