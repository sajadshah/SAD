function x = validSrtSegment(srtSegment)
    if( isempty( strfind(srtSegment.Line,' ') ) )
        x = 0;
        return ;
    end
    if ( srtSegment.Line(1) == '[' && srtSegment.Line(length(srtSegment.Line))==']' )
        x = 0;
        return ;
    end
    if( srtSegment.TimeEnd < srtSegment.TimeBegin + 1.5 )
        x = 0;
        return;
    end
    x = 1;
end