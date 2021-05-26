// ---------------------------------------------------------------------
//
// Jie Liu, Nov. 25, 2019
//



DEAL_II_NAMESPACE_OPEN

// in case we use an MPI compiler, need
// to create a communicator just for the
// current process
Timer::Timer()
  :
  start_time (0.),
  start_time_children (0.),
  start_wall_time (0.),
  cumulative_time (0.),
  cumulative_wall_time (0.),
  last_lap_time (0.),
  running (false)
#ifdef DEAL_II_WITH_MPI
  ,
  mpi_communicator (MPI_COMM_SELF),
  sync_wall_time (false)
#endif
{
#ifdef DEAL_II_WITH_MPI
  mpi_data.sum = mpi_data.min = mpi_data.max = mpi_data.avg = numbers::signaling_nan<double>();
  mpi_data.min_index = mpi_data.max_index = numbers::invalid_unsigned_int;
#endif

  start();
}


TimerOutput::~TimerOutput()
{
  try
    {
      while (active_sections.size() > 0)
        leave_subsection();
    }
  catch (...)
    {}

  if ( (output_frequency == summary || output_frequency == every_call_and_summary)
       && output_is_enabled == true)
    return_total_cpu_time();
}



double
TimerOutput::return_total_cpu_time () const
{
  // we are going to change the
  // precision and width of output
  // below. store the old values so we
  // can restore it later on
  const std::istream::fmtflags old_flags = out_stream.get_stream().flags();
  const std::streamsize    old_precision = out_stream.get_stream().precision ();
  const std::streamsize    old_width     = out_stream.get_stream().width ();

  // in case we want to write CPU times
  if (output_type != wall_times)
    {
      double total_cpu_time = Utilities::MPI::sum(timer_all(), mpi_communicator);

      // check that the sum of all times is
      // less or equal than the total
      // time. otherwise, we might have
      // generated a lot of overhead in this
      // function.
      double check_time = 0.;
      for (std::map<std::string, Section>::const_iterator
           i = sections.begin(); i!=sections.end(); ++i)
        check_time += i->second.total_cpu_time;

      const double time_gap = check_time-total_cpu_time;
      if (time_gap > 0.0)
        total_cpu_time = check_time;



      if (time_gap > 0.0)
        out_stream << std::endl
                   << "Note: The sum of counted times is " << time_gap
                   << " seconds larger than the total time.\n"
                   << "(Timer function may have introduced too much overhead, or different\n"
                   << "section timers may have run at the same time.)" << std::endl;
		  return total_cpu_time;
    }

  // restore previous precision and width
  out_stream.get_stream().precision (old_precision);
  out_stream.get_stream().width (old_width);
  out_stream.get_stream().flags (old_flags);
  return 0;
}



DEAL_II_NAMESPACE_CLOSE
